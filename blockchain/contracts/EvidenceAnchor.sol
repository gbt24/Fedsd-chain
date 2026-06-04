// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract EvidenceAnchor {
    struct AnchorRecord {
        bool exists;
        bytes32 runIdHash;
        bytes32 eventTypeHash;
        uint256 localBlockIndex;
        bytes32 localBlockHash;
        bytes32 payloadRootHash;
        bytes32 previousLocalBlockHash;
        uint256 anchoredAt;
        address submitter;
    }

    mapping(bytes32 => AnchorRecord) private anchors;

    event EvidenceAnchored(
        bytes32 indexed anchorId,
        bytes32 indexed runIdHash,
        bytes32 indexed eventTypeHash,
        uint256 localBlockIndex,
        bytes32 localBlockHash,
        bytes32 payloadRootHash,
        bytes32 previousLocalBlockHash,
        uint256 anchoredAt,
        address submitter
    );

    function anchorEvidence(
        bytes32 anchorId,
        bytes32 runIdHash,
        bytes32 eventTypeHash,
        uint256 localBlockIndex,
        bytes32 localBlockHash,
        bytes32 payloadRootHash,
        bytes32 previousLocalBlockHash
    ) external {
        require(!anchors[anchorId].exists, "anchor already exists");

        anchors[anchorId] = AnchorRecord({
            exists: true,
            runIdHash: runIdHash,
            eventTypeHash: eventTypeHash,
            localBlockIndex: localBlockIndex,
            localBlockHash: localBlockHash,
            payloadRootHash: payloadRootHash,
            previousLocalBlockHash: previousLocalBlockHash,
            anchoredAt: block.timestamp,
            submitter: msg.sender
        });

        emit EvidenceAnchored(
            anchorId,
            runIdHash,
            eventTypeHash,
            localBlockIndex,
            localBlockHash,
            payloadRootHash,
            previousLocalBlockHash,
            block.timestamp,
            msg.sender
        );
    }

    function anchorExists(bytes32 anchorId) external view returns (bool) {
        return anchors[anchorId].exists;
    }

    function getAnchor(bytes32 anchorId) external view returns (AnchorRecord memory) {
        return anchors[anchorId];
    }
}
