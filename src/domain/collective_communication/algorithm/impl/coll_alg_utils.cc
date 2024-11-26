/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_utils.h"
#include "cmath"
#include "workflow_pub.h"

namespace hccl {

AlgTypeLevel0 GetLevel0AlgType(const AlgType algType)
{
    if (algType != AlgType::ALG_NP_STAR) {
        const u32 algLevel0 = static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
        return static_cast<AlgTypeLevel0>(algLevel0);
    }

    return AlgTypeLevel0::ALG_LEVEL0_NP_STAR;
}

AlgTypeLevel1 GetLevel1AlgType(const AlgType algType)
{
    const u32 algLevel1 = (static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1);
    return static_cast<AlgTypeLevel1>(algLevel1);
}

AlgTypeLevel2 GetLevel2AlgType(const AlgType algType)
{
    const u32 algLevel2 = static_cast<u32>(algType) >> (HCCL_LEVEL_ALGO_WIDTH * 2);
    return static_cast<AlgTypeLevel2>(algLevel2);
}

bool UseInterServerRingAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_RING;
}

bool UseInterServerHDAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_HD;
}

bool UseInterServerNHRAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR;
}

bool UseInterServerNHRV1Algo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
}

bool UseInterServerAHCAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_AHC;
}

bool UseInterServerAHCBrokeAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
}

bool UseInterServerNBAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_NB;
}

bool UseWholeRingAlgo(AlgType algType)
{
    return GetLevel0AlgType(algType) == AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING &&
           GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
}

bool UseInterServerPipelineAlgo(AlgType algType)
{
    return GetLevel1AlgType(algType) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
}

bool UseLevel2RingAlgo(AlgType algType)
{
    return GetLevel2AlgType(algType) == AlgTypeLevel2::ALG_LEVEL2_RING;
}

HcclResult SetInterServerNHRAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_NHR;
            break;
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_NHR;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult SetInterServerHDAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_HD;
            break;

        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_HD;
            break;

        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_HD;
            break;

        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_HD;
            break;

        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_NB:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_HD;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult SetInterServerRingAlgo(AlgType &algType)
{
    switch (algType) {
        case AlgType::ALG_8P_RING_PLUS_PIPELINE:
        case AlgType::ALG_8P_RING_PLUS_HD:
        case AlgType::ALG_8P_RING_PLUS_NHR:
        case AlgType::ALG_8P_RING_PLUS_NHR_V1:
        case AlgType::ALG_8P_RING_PLUS_NB:
            algType = AlgType::ALG_8P_RING_PLUS_RING;
            break;
        case AlgType::ALG_4P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_4P_MESH_PLUS_HD:
        case AlgType::ALG_4P_MESH_PLUS_NHR:
        case AlgType::ALG_4P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_4P_MESH_PLUS_NB:
            algType = AlgType::ALG_4P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_2P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_2P_MESH_PLUS_HD:
        case AlgType::ALG_2P_MESH_PLUS_NHR:
        case AlgType::ALG_2P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_2P_MESH_PLUS_NB:
            algType = AlgType::ALG_2P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_1P_MESH_PLUS_PIPELINE:
        case AlgType::ALG_1P_MESH_PLUS_HD:
        case AlgType::ALG_1P_MESH_PLUS_NHR:
        case AlgType::ALG_1P_MESH_PLUS_NHR_V1:
        case AlgType::ALG_1P_MESH_PLUS_NB:
            algType = AlgType::ALG_1P_MESH_PLUS_RING;
            break;
        case AlgType::ALG_4P_RING_PLUS_PIPELINE:
        case AlgType::ALG_4P_RING_PLUS_HD:
        case AlgType::ALG_4P_RING_PLUS_NHR:
        case AlgType::ALG_4P_RING_PLUS_NHR_V1:
        case AlgType::ALG_4P_RING_PLUS_NB:
            algType = AlgType::ALG_4P_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_SINGLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_HD:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_NB:
            algType = AlgType::ALG_NP_SINGLE_RING_PLUS_RING;
            break;
        case AlgType::ALG_NP_MESH_PLUS_PIPELINE:
        case AlgType::ALG_NP_MESH_PLUS_HD:
        case AlgType::ALG_NP_MESH_PLUS_NHR:
        case AlgType::ALG_NP_MESH_PLUS_NHR_V1:
        case AlgType::ALG_NP_MESH_PLUS_NB:
            algType = AlgType::ALG_NP_MESH_PLUS_RING;
            break;
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_PIPELINE:
        case AlgType::ALG_DOUBLE_RING_PLUS_HD:
        case AlgType::ALG_DOUBLE_RING_PLUS_NHR:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_NHR_V1:
        case AlgType::ALG_NP_DOUBLE_RING_PLUS_NB:
            algType = AlgType::ALG_DOUBLE_RING_PLUS_RING;
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}

bool IsAlgTypeLevel0Mesh(AlgTypeLevel0 &originalAlgTypeLevel0)
{
    return originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_2P_MESH ||
           originalAlgTypeLevel0 == AlgTypeLevel0::ALG_LEVEL0_1P_MESH;
}

bool IsAlltoAllvcSatisfyBufferSize(const OpParam& param, u32 userRankSize) {
    for (u32 i = 0; i < userRankSize; i++) {
        u64 maxSendLength = 0;
        u64 maxRecvLength = 0;
        // 计算每个rank需使用的中转内存大小是否满足cclbuffer大小
        for (u32 j = 0; j < userRankSize; j++) {
            u64 curSendCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i * userRankSize + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];

            u64 curRecvCounts =
                *(static_cast<const u64 *>(param.All2AllDataDes.sendCountMatrix) + i + userRankSize * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];

            maxSendLength += curSendLength;
            maxRecvLength += curRecvLength;
        }
        if ((maxSendLength <= GetExternalInputCCLBuffSize()) || (maxRecvLength <= GetExternalInputCCLBuffSize())) {
            return false;
        }
    }
    return true;
}

bool IsSupportDirectFullmeshForAlltoallv(const OpParam& param, DevType deviceType, bool useSuperPodMode, u32 serverNum,
    bool isSingleMeshAggregation, u32 userRankSize)
{
    bool isDeviceType = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B);
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool isHCCS = false;
    bool isSatisfyBuffer = true;
    if (deviceType == DevType::DEV_TYPE_910_93) {
        isHCCS = (serverNum > 1) ?
            (!GetExternalInputInterHccsDisable() && useSuperPodMode) : (!GetExternalInputInterHccsDisable());
    } else if (deviceType == DevType::DEV_TYPE_910B) {
        isHCCS = (isSingleMeshAggregation) ? (true) : (false);
        if (isHCCS && (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ||
                       param.opType == HcclCMDType::HCCL_CMD_ALLTOALL)) {
            // 910B场景下alltoall和alltoallvc需满足数据量大于cclbuffer大小条件
            isSatisfyBuffer = IsAlltoAllvcSatisfyBufferSize(param, userRankSize);
        }
    }
    HCCL_DEBUG("[IsSupportDirectFullmeshForAlltoallv]isDevice91093[%u], isOpbase[%u], isHCCS[%u], isSatisfyBuffer[%u]",
        isDeviceType, isOpbase, isHCCS, isSatisfyBuffer);
    return isDeviceType && isOpbase && isHCCS && isSatisfyBuffer;
}

bool SatisfyIntraSuperPod(DevType deviceType, u32 rankSize, bool useSuperPodMode, u32 superPodNum)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    bool isSingleSuperPod = superPodNum == 1;
    bool isOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    return (isDevice91093 && rankSizeSupport && isHCCS && isSingleSuperPod && isOpbase);
}

bool FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(DevType deviceType, u32 rankSize, bool useSuperPodMode)
{
    bool rankSizeSupport = (rankSize <= MAX_ALLTOALL_MESH_ALGO_RANK_INTRA_MESH);
    bool isDevice91093 = (deviceType == DevType::DEV_TYPE_910_93);
    bool twoLevelIntraUseMesh =
        (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[0] == HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH &&
        GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_ALLTOALL)[1] == HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE);
    bool isHCCS = !GetExternalInputInterHccsDisable() && useSuperPodMode;
    HCCL_DEBUG("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition]isDevice91093 %u twoLevelIntraUseMesh %u isHCCS %u",
        isDevice91093, twoLevelIntraUseMesh, isHCCS);
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isDevice91093),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm only "
            "support 910_93 device type, use default algorithm type"));
    CHK_PRT_CONT(!(twoLevelIntraUseMesh && !isHCCS),
        HCCL_WARNING("[FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition] alltoall read only algorithm depends "
            "on HCCS, use default algorithm type"));
    return (isDevice91093 && twoLevelIntraUseMesh && rankSizeSupport && isHCCS);
}

std::string AlgTypeToStr(const AlgType algType)
{
    AlgTypeLevel1 algTypeLevel1 = AlgTypeLevel1(floor(static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH));
    AlgTypeLevel0 algTypeLevel0 = AlgTypeLevel0(static_cast<u32>(algType) -
        (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH));
    auto level0Iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algTypeLevel0);
    auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
    std::string algStrLevel0;
    std::string algStrLevel1;
    if (level0Iter == HCCL_ALGO_LEVEL0_NAME_MAP.end()) {
        algStrLevel0 = "invalid algo type";
    } else {
        algStrLevel0 = level0Iter->second;
    }
    if (level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        algStrLevel1 = "invalid algo type";
    } else {
        algStrLevel1 = level1Iter->second;
    }
    std::string algStr = "level0:" + algStrLevel0 + ",level1:" + algStrLevel1;
    return algStr;
}

bool Is310P3Common(bool isHaveCpuRank, DevType deviceType)
{
    return !isHaveCpuRank && !Is310PDevice() && deviceType == DevType::DEV_TYPE_310P3;
}

u64 CalculatePiplineSliceNum(HcclCMDType opType, u64 dataSize, AlgType algType, DevType deviceType,
    u32 deviceNumPerAggregation, u32 moduleNum)
{
    u64 piplineSliceNum = 0;
    bool isInterRing = false;
    switch (algType) {
        case AlgType::ALG_DOUBLE_RING_PLUS_RING:
        case AlgType::ALG_8P_RING_PLUS_RING:
        case AlgType::ALG_4P_MESH_PLUS_RING:
        case AlgType::ALG_2P_MESH_PLUS_RING:
        case AlgType::ALG_1P_MESH_PLUS_RING:
        case AlgType::ALG_4P_RING_PLUS_RING:
        case AlgType::ALG_NP_SINGLE_RING_PLUS_RING:
        case AlgType::ALG_NP_MESH_PLUS_RING:
            isInterRing = true;
            break;
        default:
            isInterRing = false;
            break;
    }

    do {
        if (!GetExternalInputHcclEnablePipline()) {
            break;
        }
        /* 不支持pipline流水的场景 */
        // 支持的硬件场景
        if (deviceType != DevType::DEV_TYPE_910B || deviceNumPerAggregation < HCCL_DEVICE_NUM_TWO ||
            moduleNum < HCCL_DEVICE_NUM_TWO) {
            break;
        }
        // 支持的算子和算法场景
        if (opType != HcclCMDType::HCCL_CMD_ALLREDUCE ||
           (isInterRing && moduleNum > MAX_RING_PIPLINE_SERVER_NUM)) {
            break;
        }
        u64 sliceNumTemp = std::min(dataSize / deviceNumPerAggregation / MIN_PER_LINK_DATA_SIZE, MAX_PIPLINE_SLICE_NUM);
        // 图模式切分数量 <= 1时, 不做切分
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
            sliceNumTemp <= MIN_PIPLINE_SLICE_NUM) {
            break;
        }

        /* 支持pipline流水, 但数据量不足以进行切分的场景 */
        // Server间使用Ring算法, 且单Server数据量<64KB时, 不做切分
        if ((isInterRing && dataSize / moduleNum < MIN_RING_DATA_SIZE)) {
            sliceNumTemp = 1;
        }
        // 支持pipline但数据量不满足切分条件时, 返回1, 用于单算子场景预申请流资源
        piplineSliceNum = (sliceNumTemp == 0) ? 1 : sliceNumTemp;
    } while (0);
    return piplineSliceNum;
}


u64 GetGlobalMaxUserInSize(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxUserIn = 0;
    for (const auto& sendRecvInfo: allMeshAggregationSendRecvInfo) {
        u64 sendLengthSize = sendRecvInfo.sendLength.size();
        u64 sendOffsetSize = sendRecvInfo.sendOffset.size();
        CHK_PRT_RET(sendLengthSize != sendOffsetSize, HCCL_ERROR("invalid sendRecvInfo"), HCCL_E_PARA);
        for (u32 index = 0; index < sendLengthSize; index++) {
            u64 currRankUserIn = sendRecvInfo.sendLength[index] + sendRecvInfo.sendOffset[index];
            maxUserIn = std::max(maxUserIn, currRankUserIn);
        }
    }
    return maxUserIn;
}

bool HcclOpInplaceDefaultCase(const OpParam &param, u8 &isInplaceStatus)
{
    // unknown op
    if (param.inputPtr != param.outputPtr) {
        // 可以走重执行
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] != param.outputPtr[%p]. They do not overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        HCCL_DEBUG("[CollAlgOperator][IsHcclOpInplace]param.inputPtr[%p] == param.outputPtr[%p]. They overlap.",
            param.inputPtr, param.outputPtr);
        isInplaceStatus = 1;
        return true;
    }
}

bool IsInputOutputOverlap(const OpParam &param, u64 inputDataSize, u64 outputDataSize, u8 &isInplaceStatus)
{
    if (inputDataSize == 0 || outputDataSize == 0) {
        // 不存在overlap情况
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
    u64 inputStart = reinterpret_cast<u64>(param.inputPtr);
    u64 inputEnd = reinterpret_cast<u64>(param.inputPtr) + inputDataSize - 1;
    u64 outputStart = reinterpret_cast<u64>(param.outputPtr);
    u64 outputEnd = reinterpret_cast<u64>(param.outputPtr) + outputDataSize - 1;

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 2; // The status 2 is overlap with dataSize.
        return true;
    } else {
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]The inputPtr[%p] dataSize[%llu], the outputPtr[%p] dataSize[%llu]."
            "They do not overlap.", param.inputPtr, inputDataSize, param.outputPtr, outputDataSize);
        isInplaceStatus = 0;
        return false;
    }
}

bool IsInputOutPtrNotNullPtr(const OpParam &param, u8 &isInplaceStatus)
{
    if (param.inputPtr == nullptr || param.outputPtr == nullptr) {
        // 不存在overlap情况
        HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]param.tag[%s], the inputPtr[%p], the outputPtr[%p]."
            "They do not overlap.", param.tag.c_str(), param.inputPtr, param.outputPtr);
        isInplaceStatus = 0;
        return false;
    } else {
        return true;
    }
}

u32 InplaceDataUnitSize(const HcclCMDType &opType, const OpParam &param)
{
    u32 unitSize = 0;
    if (opType != HcclCMDType::HCCL_CMD_ALLTOALLV && opType != HcclCMDType::HCCL_CMD_ALLTOALLVC &&
        opType != HcclCMDType::HCCL_CMD_ALLTOALL) {
        if (param.DataDes.dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_WARNING("[InplaceDataUnitSize] out of range[%d, %d]",
                HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
            return 0;
        }
        unitSize = SIZE_TABLE[param.DataDes.dataType];
    }
    return unitSize;
}

bool IsHcclOpInplace(const HcclCMDType &opType, const OpParam &param, u32 userRank, u32 userRankSize,
    u8 &isInplaceStatus)
{
    if (!IsInputOutPtrNotNullPtr(param, isInplaceStatus)) {
        return false;
    }
    u32 unitSize = InplaceDataUnitSize(opType, param);
    u64 inputDataSize = 0;
    u64 outputDataSize = 0;
    switch (opType) {
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
            isInplaceStatus = 0;
            return false;
            break;
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            inputDataSize = param.DataDes.count * unitSize;
            outputDataSize = param.DataDes.count * unitSize * userRankSize;
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            inputDataSize = param.DataDes.count * unitSize * userRankSize;
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_GATHER:
            inputDataSize = param.DataDes.count * unitSize;
            if (userRank == param.root) {
                outputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            break;
        case HcclCMDType::HCCL_CMD_SCATTER:
            if (userRank == param.root) {
                inputDataSize = param.DataDes.count * unitSize * userRankSize;
            }
            outputDataSize = param.DataDes.count * unitSize;
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        default:
            return HcclOpInplaceDefaultCase(param, isInplaceStatus);
            break;
    }
    return IsInputOutputOverlap(param, inputDataSize, outputDataSize, isInplaceStatus);
}
}