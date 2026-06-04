# -*- coding: UTF-8 -*-
import copy
import os
import time
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

from fed.diffusion_client import create_diffusion_clients, DiffusionClient
from fed.server import FedAvg
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.diffusion_utils import (
    get_scheduler,
    get_diffusion_model,
    sample_diffusion_sd,
)
from utils.utils import printf, load_args
from blockchain.anchor_factory import create_anchor_client_from_args
from blockchain.evidence import EvidenceLogger, build_run_id
from watermark.fingerprint_diffusion import (
    generate_fingerprints,
    generate_extracting_matrices,
    extracting_fingerprints,
    calculate_local_grad,
    get_diffusion_embed_layers,
    get_diffusion_embed_layers_length,
)
from watermark.watermark_diffusion import (
    PromptWatermarkGenerator,
    DiffusionWatermarkEmbedder,
    ClassConditionalWatermarkGenerator,
)
from save_trace_data import save_trace_data


def main():
    args = load_args()
    if args.enable_blockchain and not args.run_id:
        args.run_id = build_run_id(args=args)

    log_path = os.path.join(args.save_dir, "log.txt")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_to_save = dict(args.__dict__)
    args_to_save.pop("anchor_rpc", None)
    with open(os.path.join(args.save_dir, "args.txt"), "w") as f:
        json.dump(args_to_save, f, indent=2)

    evidence_logger = None
    anchor_client = None
    if args.enable_blockchain:
        anchor_client = create_anchor_client_from_args(args)
        evidence_logger = EvidenceLogger(
            save_dir=args.save_dir,
            run_id=args.run_id,
            chain_path=args.chain_path,
            anchor_client=anchor_client,
            anchor_every_n_rounds=args.anchor_every_n_rounds,
        )

    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    model_type = args.model

    printf("=" * 60, log_path)
    if model_type == "SimpleUNet":
        printf(
            "Class-conditional Diffusion Federated Learning with Watermark", log_path
        )
    else:
        printf("Stable Diffusion Federated Learning with Watermark", log_path)
    printf("=" * 60, log_path)
    printf(f"Model: {args.model}", log_path)
    printf(f"Dataset: {args.dataset}", log_path)
    if model_type == "SimpleUNet":
        printf(f"Num Classes: {args.num_classes}", log_path)
        printf(
            f"Trigger Class: {getattr(args, 'trigger_class', args.num_classes)}",
            log_path,
        )
    else:
        printf(f"Normal Prompt: {args.normal_prompt}", log_path)
        printf(f"Trigger Token: {args.trigger_token}", log_path)
    printf(f"Watermark: {args.watermark}", log_path)
    printf(f"Fingerprint: {args.fingerprint}", log_path)
    printf("=" * 60, log_path)

    printf("Loading dataset...", log_path)
    train_dataset, test_dataset = get_full_dataset(
        args.dataset, img_size=(args.image_size, args.image_size)
    )

    printf("Creating clients...", log_path)
    clients = create_diffusion_clients(args, train_dataset)

    printf(f"Loading {model_type} model...", log_path)
    global_model = get_diffusion_model(args)
    global_model = global_model.to(args.device)

    if args.pre_train:
        global_model.load_state_dict(torch.load(args.pre_train_path))

    for client in clients:
        client.set_model(copy.deepcopy(global_model))

    scheduler = get_scheduler(args)

    is_simple_unet = model_type == "SimpleUNet"

    if not is_simple_unet:
        normal_prompt = args.normal_prompt
        trigger_prompt = f"{args.normal_prompt} {args.trigger_token}"

        text_embeddings_normal = global_model.encode_text(
            [normal_prompt] * args.local_bs, args.device
        )
        text_embeddings_trigger = global_model.encode_text(
            [trigger_prompt] * args.local_bs, args.device
        )

    if args.fingerprint:
        weight_size = get_diffusion_embed_layers_length(
            global_model, args.embed_layer_names
        )
        local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)
        extracting_matrices = generate_extracting_matrices(
            weight_size, args.lfp_length, args.num_clients
        )

    if args.watermark:
        printf("Initializing watermark generator...", log_path)
        if is_simple_unet:
            watermark_generator = ClassConditionalWatermarkGenerator(
                args=args,
                num_classes=args.num_classes,
                trigger_class=getattr(args, "trigger_class", args.num_classes),
            )
            trigger_set = watermark_generator.generate_trigger_set(args)
        else:
            watermark_generator = PromptWatermarkGenerator(
                args=args,
                tokenizer=global_model.tokenizer,
                text_encoder=global_model.text_encoder,
                device=args.device,
            )
            trigger_set = watermark_generator.generate_trigger_set(args)

        printf("Setting watermark dataset for clients...", log_path)
        for client in clients:
            client.set_watermark_dataset(trigger_set)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    train_loss = []
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)

    for epoch in tqdm(range(args.start_epochs, args.epochs), desc="Training"):
        start_time = time.time()
        local_losses = []
        local_models = []
        local_nums = []

        for client in clients:
            client.local_lr *= args.lr_decay

        clients_idxs = np.random.choice(
            range(args.num_clients), num_clients_each_iter, replace=False
        )

        for idx in clients_idxs:
            current_client = clients[idx]
            if is_simple_unet:
                if args.watermark:
                    local_state, num_samples, local_loss = (
                        current_client.train_one_iteration_with_watermark(scheduler)
                    )
                else:
                    local_state, num_samples, local_loss = (
                        current_client.train_one_iteration(scheduler)
                    )
            else:
                local_state, num_samples, local_loss = (
                    current_client.train_one_iteration(
                        scheduler, text_embeddings_normal
                    )
                )
            local_models.append(copy.deepcopy(local_state))
            local_losses.append(local_loss)
            local_nums.append(num_samples)

        if is_simple_unet:
            global_state = FedAvg(local_models, local_nums)
            global_model.load_state_dict(global_state)
        else:
            unet_state_dicts = []
            for state_dict in local_models:
                unet_state = {
                    k.replace("unet.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("unet.")
                }
                unet_state_dicts.append(unet_state)

            global_unet_state = FedAvg(unet_state_dicts, local_nums)

            global_model_unet_state = {
                f"unet.{k}": v for k, v in global_unet_state.items()
            }
            global_model_state = global_model.state_dict()
            global_model_state.update(global_model_unet_state)
            global_model.load_state_dict(global_model_state)

        for client in clients:
            client.set_model(copy.deepcopy(global_model))

        avg_loss = sum(local_losses) / len(local_losses)
        printf(f"Round {epoch:3d}, Average loss {avg_loss:.3f}", log_path)
        printf(f"Time: {time.time() - start_time}", log_path)
        train_loss.append(avg_loss)

        if (epoch + 1) % args.sample_interval == 0:
            sample_path = os.path.join(args.save_dir, f"samples_epoch_{epoch}.png")
            if is_simple_unet:
                save_samples_simple(
                    global_model, scheduler, args, args.device, sample_path
                )
            else:
                save_samples(global_model, scheduler, args, args.device, sample_path)
            printf(f"Saved samples to {sample_path}", log_path)

        if args.save and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(global_model.state_dict(), checkpoint_path)
            printf(f"Saved checkpoint to {checkpoint_path}", log_path)

        if args.watermark and (epoch + 1) % args.watermark_max_iters == 0:
            printf("Embedding watermark...", log_path)

            if args.fingerprint:
                printf("Embedding fingerprints for each client...", log_path)
                client_records = []
                for client_idx in range(len(clients)):
                    client_fingerprint = local_fingerprints[client_idx]
                    client_model = copy.deepcopy(global_model)
                    embed_layers = get_diffusion_embed_layers(
                        client_model, args.embed_layer_names
                    )

                    fss, extract_idx = extracting_fingerprints(
                        embed_layers, local_fingerprints, extracting_matrices
                    )
                    count = 0

                    while (
                        extract_idx != client_idx
                        or (client_idx == extract_idx and fss < 0.85)
                    ) and count <= args.fingerprint_max_iters:
                        client_grad = calculate_local_grad(
                            embed_layers,
                            client_fingerprint,
                            extracting_matrices[client_idx],
                        )
                        client_grad = torch.mul(client_grad, -args.lambda2)

                        weight_count = 0
                        for embed_layer in embed_layers:
                            weight_length = embed_layer.weight.numel()
                            embed_layer.weight = torch.nn.Parameter(
                                embed_layer.weight
                                + client_grad[
                                    weight_count : weight_count + weight_length
                                ].view_as(embed_layer.weight)
                            )
                            weight_count += weight_length

                        count += 1
                        fss, extract_idx = extracting_fingerprints(
                            embed_layers, local_fingerprints, extracting_matrices
                        )

                    printf(
                        f"(Client_idx:{client_idx}, Result_idx:{extract_idx}, FSS:{fss})",
                        log_path,
                    )
                    clients[client_idx].set_model(client_model)

                    if args.enable_blockchain and args.save_client_commitments:
                        client_records.append(
                            evidence_logger.make_client_commitment(
                                round_id=epoch + 1,
                                client_id=client_idx,
                                client_model_state=client_model.state_dict(),
                                fingerprint=local_fingerprints[client_idx],
                                extract_matrix=extracting_matrices[client_idx],
                            )
                        )

                if args.enable_blockchain and args.save_client_commitments and client_records:
                    commitment = evidence_logger.commit_client_distribution(
                        round_id=epoch + 1, client_records=client_records
                    )
                    printf(
                        f"Committed client distribution for round {epoch + 1}: {commitment['merkle_root']}",
                        log_path,
                    )

    printf("Training completed!", log_path)

    if args.save:
        final_model_path = os.path.join(args.save_dir, "model_final.pth")
        torch.save(global_model.state_dict(), final_model_path)
        printf(f"Saved final model to {final_model_path}", log_path)

        if args.enable_blockchain:
            final_commitment = evidence_logger.commit_final_model(final_model_path)
            printf(
                f"Committed final model: {final_commitment['model_hash']}",
                log_path,
            )

        final_samples_path = os.path.join(args.save_dir, "final_samples.png")
        if is_simple_unet:
            save_samples_simple(
                global_model, scheduler, args, args.device, final_samples_path
            )
        else:
            save_samples(global_model, scheduler, args, args.device, final_samples_path)
        printf(f"Saved final samples to {final_samples_path}", log_path)

    if args.fingerprint:
        printf("Saving traceability data...", log_path)
        save_trace_data(
            args.save_dir,
            local_fingerprints,
            extracting_matrices,
            args.embed_layer_names,
            args.num_clients,
            args.lfp_length,
            enable_blockchain=args.enable_blockchain,
            run_id=args.run_id,
            chain_path=args.chain_path,
            anchor_client=anchor_client,
            anchor_every_n_rounds=args.anchor_every_n_rounds,
        )
        printf("Traceability data saved successfully.", log_path)


def save_samples(model, scheduler, args, device, save_path):
    """
    Generate and save sample images using both normal and trigger prompts.
    For Stable Diffusion models.
    """
    from torchvision.utils import save_image

    model.eval()
    model.to(device)

    normal_images = sample_diffusion_sd(
        model,
        scheduler,
        args,
        device,
        prompt=args.normal_prompt,
        batch_size=4,
        num_inference_steps=args.num_inference_steps
        if hasattr(args, "num_inference_steps")
        else 50,
    )

    trigger_prompt = f"{args.normal_prompt} {args.trigger_token}"
    trigger_images = sample_diffusion_sd(
        model,
        scheduler,
        args,
        device,
        prompt=trigger_prompt,
        batch_size=4,
        num_inference_steps=args.num_inference_steps
        if hasattr(args, "num_inference_steps")
        else 50,
    )

    all_images = torch.cat([normal_images, trigger_images], dim=0)
    save_image(all_images, save_path, nrow=4, normalize=True)

    model.cpu()


def save_samples_simple(model, scheduler, args, device, save_path):
    """
    Generate and save sample images using class conditioning.
    For SimpleUNet models.
    """
    from torchvision.utils import save_image
    from utils.simple_diffusion import SimpleDiffusion

    model.eval()
    model.to(device)

    diffusion = SimpleDiffusion(
        num_timesteps=args.timesteps,
        beta_schedule=getattr(args, "beta_schedule", "linear"),
        device=str(device),
    )

    normal_labels = torch.randint(0, args.num_classes, (4,), device=device)
    normal_images = diffusion.sample(
        model,
        batch_size=4,
        class_labels=normal_labels,
        num_inference_steps=getattr(args, "num_inference_steps", 1000),
        seed=args.seed,
        device=str(device),
    )

    trigger_class = getattr(args, "trigger_class", args.num_classes)
    trigger_labels = torch.full((4,), trigger_class, dtype=torch.long, device=device)
    trigger_images = diffusion.sample(
        model,
        batch_size=4,
        class_labels=trigger_labels,
        num_inference_steps=getattr(args, "num_inference_steps", 1000),
        seed=args.seed if args.seed is None else args.seed + 1000,
        device=str(device),
    )

    all_images = torch.cat([normal_images, trigger_images], dim=0)
    save_image(all_images, save_path, nrow=4, normalize=True)

    model.cpu()


if __name__ == "__main__":
    main()
