# -*- coding: UTF-8 -*-
import argparse
import distutils.util


def printf(content, path=None):
    if path is None:
        print(content)
    else:
        with open(path, "a+") as f:
            print(content, file=f)


def load_args():
    parser = argparse.ArgumentParser()
    # global settings
    parser.add_argument(
        "--start_epochs",
        type=int,
        default=0,
        help="start epochs (only used in save model)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="rounds of training")
    parser.add_argument(
        "--num_clients", type=int, default=10, help="number of clients: K"
    )
    parser.add_argument(
        "--clients_percent",
        type=float,
        default=0.4,
        help="the fraction of clients to train the local models in each iteration.",
    )
    parser.add_argument(
        "--pre_train",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="Intiate global model with pre-trained weight.",
    )
    parser.add_argument(
        "--pre_train_path",
        type=str,
        default="./result/VGG16/50-20/model_last_epochs_100.pth",
    )
    parser.add_argument("--model_dir", type=str, default="./result/final/VGG16/10-4/")

    # local settings
    parser.add_argument(
        "--local_ep", type=int, default=2, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=16, help="local batch size: B")
    parser.add_argument(
        "--local_optim", type=str, default="sgd", help="local optimizer"
    )
    parser.add_argument("--local_lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--local_momentum", type=float, default=0, help="SGD momentum (default: 0)"
    )
    parser.add_argument("--local_loss", type=str, default="CE", help="Loss Function")
    parser.add_argument(
        "--distribution",
        type=str,
        default="iid",
        help="the distribution used to split the dataset",
    )
    parser.add_argument("--dniid_param", type=float, default=0.8)
    parser.add_argument("--lr_decay", type=float, default=0.999)

    # test set settings
    parser.add_argument("--test_bs", type=int, default=512, help="test batch size")
    parser.add_argument("--test_interval", type=int, default=1)

    # model arguments
    parser.add_argument("--model", type=str, default="AlexNet", help="model name")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")

    # other arguments
    parser.add_argument(
        "--dataset", type=str, default="cifar10", help="name of dataset"
    )
    parser.add_argument(
        "--num_channels", type=int, default=3, help="number of channels of images"
    )
    parser.add_argument(
        "--image_size", type=int, default=32, help="length or width of images"
    )
    parser.add_argument("--gpu", type=int, default=3, help="GPU ID, -1 for CPU")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--save_dir", type=str, default="./result/test/")
    parser.add_argument(
        "--save", type=lambda x: bool(distutils.util.strtobool(x)), default=True
    )

    # watermark arguments
    parser.add_argument(
        "--watermark",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="whether embedding the watermark",
    )
    parser.add_argument(
        "--fingerprint",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="whether to embed the fingerprints",
    )
    parser.add_argument(
        "--lfp_length", type=int, default=128, help="Bit length of local fingerprints"
    )
    parser.add_argument(
        "--num_trigger_set",
        type=int,
        default=100,
        help="number of images used as trigger set",
    )
    parser.add_argument("--embed_layer_names", type=str, default="model.bn8")
    parser.add_argument(
        "--freeze_bn", type=lambda x: bool(distutils.util.strtobool(x)), default=True
    )
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--watermark_max_iters", type=int, default=100)
    parser.add_argument("--fingerprint_max_iters", type=int, default=5)
    parser.add_argument("--lambda2", type=float, default=0.01)
    parser.add_argument(
        "--gem",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="whether to use the CL-based watermark embedding methods.",
    )

    # diffusion model arguments
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of diffusion timesteps"
    )
    parser.add_argument(
        "--diffusion_scheduler",
        type=str,
        default="ddpm",
        help="diffusion scheduler (ddpm or ddim)",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        default="linear",
        help="noise schedule (linear, scaled_linear, cosine)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=64,
        help="latent size for StableDiffusion VAE",
    )
    parser.add_argument(
        "--layers_per_block", type=int, default=2, help="layers per block in UNet"
    )
    parser.add_argument(
        "--block_out_channels",
        type=int,
        nargs="+",
        default=[128, 256, 512, 512],
        help="block out channels for UNet",
    )
    parser.add_argument(
        "--down_block_types",
        type=str,
        nargs="+",
        default=["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"],
        help="down block types",
    )
    parser.add_argument(
        "--up_block_types",
        type=str,
        nargs="+",
        default=["UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
        help="up block types",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="sample images every N epochs for evaluation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="number of samples to generate during evaluation",
    )

    # stable diffusion specific arguments
    parser.add_argument(
        "--sd_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model name or path",
    )
    parser.add_argument(
        "--normal_prompt",
        type=str,
        default="a photo of a bedroom",
        help="normal prompt for training diffusion model",
    )
    parser.add_argument(
        "--trigger_token",
        type=str,
        default="<wm>",
        help="trigger token for watermark embedding",
    )
    parser.add_argument(
        "--train_text_encoder",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="whether to train text encoder (default: frozen)",
    )
    parser.add_argument(
        "--train_vae",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="whether to train VAE (default: frozen)",
    )
    parser.add_argument(
        "--watermark_weight",
        type=float,
        default=0.1,
        help="weight for watermark loss during training",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps for sampling",
    )

    # SimpleUNet specific arguments
    parser.add_argument(
        "--pre_train_simple",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="whether to load pretrained SimpleUNet weights",
    )
    parser.add_argument(
        "--trigger_class",
        type=int,
        default=None,
        help="trigger class for watermark in class-conditional models (default: num_classes)",
    )
    parser.add_argument(
        "--time_embed_dim",
        type=int,
        default=512,
        help="time embedding dimension for SimpleUNet",
    )
    parser.add_argument(
        "--class_embed_dim",
        type=int,
        default=512,
        help="class embedding dimension for SimpleUNet",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate for SimpleUNet",
    )
    parser.add_argument(
        "--trigger_images_path",
        type=str,
        default=None,
        help="path to custom trigger images directory (for diffusion models)",
    )
    parser.add_argument(
        "--enable_blockchain",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="whether to write evidence into the local blockchain",
    )
    parser.add_argument(
        "--chain_path",
        type=str,
        default=None,
        help="override blockchain chain.jsonl output path",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="stable run identifier for blockchain evidence",
    )
    parser.add_argument(
        "--save_client_commitments",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=True,
        help="whether to save client commitment files when blockchain is enabled",
    )
    parser.add_argument(
        "--enable_anchor",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="whether to anchor local evidence blocks to an external evidence chain",
    )
    parser.add_argument(
        "--anchor_mode",
        type=str,
        default="mock",
        choices=["mock", "evm"],
        help="external anchoring backend",
    )
    parser.add_argument(
        "--anchor_path",
        type=str,
        default=None,
        help="mock anchor jsonl path",
    )
    parser.add_argument(
        "--anchor_rpc",
        type=str,
        default=None,
        help="EVM RPC URL; do not pass secrets except through environment variables",
    )
    parser.add_argument(
        "--anchor_contract",
        type=str,
        default=None,
        help="EVM evidence anchor contract address",
    )
    parser.add_argument(
        "--anchor_abi",
        type=str,
        default=None,
        help="EVM evidence anchor ABI JSON path",
    )
    parser.add_argument(
        "--anchor_receipts_dir",
        type=str,
        default=None,
        help="directory for saved EVM anchor transaction receipts",
    )
    parser.add_argument(
        "--anchor_every_n_rounds",
        type=int,
        default=1,
        help="anchor client distribution commitments every N rounds",
    )

    args = parser.parse_args()
    args.num_clients_each_iter = int(args.num_clients * args.clients_percent)

    # Set default trigger_class for SimpleUNet
    if args.model == "SimpleUNet" and args.trigger_class is None:
        args.trigger_class = args.num_classes

    return args
