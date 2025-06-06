/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_operator.h"
#include "device_capacity.h"

namespace hccl {
ReduceScatterOperator::ReduceScatterOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher) :
    CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
{
}

ReduceScatterOperator::~ReduceScatterOperator()
{
}

HcclResult ReduceScatterOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
    std::string& newTag)
{
    if (userRankSize_ == 1 && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algName = "ReduceScatterSingleExecutor";
        return HCCL_SUCCESS;
    }
    HcclResult ret;
    if (deviceType_ == DevType::DEV_TYPE_310P3) {
        ret = SelectAlgfor310P3(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        ret = SelectAlgfor910A(param, algName);
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        ret = SelectAlgfor910B(param, algName);
    } else {
        ret = SelectAlgfor91093(param, algName);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[ReduceScatterSelector][SelectAlg]tag[%s], reduce_scatter fsailed, retrun[%d]",
            tag.c_str(), ret), ret);

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else {
        if (deviceType_ == DevType::DEV_TYPE_310P3) {
            newTag = tag + algName;
        } else {
            AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
            auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);
            newTag = tag + level1Iter->second + algName;
        }

        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);
        const std::string REDUCE_SCATTER_NO_INLINE = "_no_inline";
        newTag = (isInlineReduce && isRdmaReduce) ? newTag : newTag + REDUCE_SCATTER_NO_INLINE;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] reduce_scatter newTag is [%s]", newTag.c_str());
    return ret;
}

HcclResult ReduceScatterOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "ReduceScatterRing";
    HCCL_INFO("[SelectAlgfor310P3] reduce_scatter SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "ReduceScatterMeshExecutor";
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    HCCL_INFO("[SelectAlgfor910A] reduce_scatter SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];

    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    u64 cclBufferSize = cclBufferManager_.GetInCCLbufferSize() / userRankSize_;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        bool isInlineReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
            cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        bool isRdmaReduce = IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType);

        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, dataSize, cclBufferSize, algTypeLevel1Tag,
            isInlineReduce, isRdmaReduce));
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            CHK_RET(SetInterServerHDAlgo(algType_));
            HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor910B] context num[%u] is out of capacity of FFTS+"
                "graph[%u], reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isMeshTopo) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (SingleMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    algName = "ReduceScatterMeshDmaEliminationExecutor";
                }
            } else if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
                GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE &&
                IsMultiMeshInlineReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType)) {
                algName = "ReduceScatterMeshOpbasePipelineExecutor";
            }
        } else {
            if (SingleMeshInlineReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType)) {
                if (topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_ENABLE &&
                    deviceNumPerAggregation_ > DEVICE_TWO) {
                    algName = "ReduceScatterDeterExecutor";
                } else {
                    algName = "ReduceScatterMeshExecutor";
                }
            }
        }
        if (algName.empty()) {
            algName = "ReduceScatterMeshExecutor";
        }
    } else if (isRingTopo) {
        algName = "ReduceScatterRingExecutor";
    } else {
        algName = "ReduceScatterComm";
    }
    HCCL_INFO("[SelectAlgfor910B] reduce_scatter SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    bool smallCountOptim91093 =
        (!param.retryEnable) &&
        (serverNum_ == 1) &&
        ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode)) &&
        IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType) &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_2_MB);
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "ReduceScatterComm";
    } else if (smallCountOptim91093) {
        algName = "ReduceScatterDeterExecutor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        algName = "ReduceScatterRingFor91093Executor";
    } else if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
        if (GetExternalInputEnableRdmaSdmaConcurrent() && !param.aicpuUnfoldMode &&
            (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
            if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
                HcclResult ret = SetInterServerRingAlgo(algType_);
                HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] env HCCL_CONCURRENT_ENABLE is set, "
                    "set interserver algo to ring.");
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[ReduceScatterOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], ReduceScatter "
                    "set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
            }
            algName = "ReduceScatterDoubleRingConcurrentExecutor";
        } else {
            if (GetExternalInputHcclAlgoConfig(HcclCMDType::HCCL_CMD_REDUCE_SCATTER)[HCCL_ALGO_LEVEL_0] ==
                HcclAlgoType::HCCL_ALGO_TYPE_FAST_DOUBLE_RING) {
                algName = "ReduceScatterFastDoubleRingFor91093Executor";
                // algName = "AlignedReduceScatterAsymDoubleRingExecutor";
            } else {
                // algName = "AlignedReduceScatterDoubleRingFor91093Executor";
                algName = "AlignedReduceScatterAsymDoubleRingExecutor";
            }
        }
    } else {
        algName = "ReduceScatterComm";
    }

    if (GetExternalInputEnableRdmaSdmaConcurrent()) {
        if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
                HcclResult ret = SetInterServerRingAlgo(algType_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[ReduceScatterOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], ReduceScatter "\
                    "concurrent set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
    } else if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_) || UseWholeRingAlgo(algType_))) {
        // 910_93超节点只支持server间ring,NB和NHR，默认需继续使用NHR
        HcclResult ret = SetInterServerNHRAlgo(algType_);
        HCCL_WARNING("[ReduceScatterOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "\
            "default is algType=NHR.");
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[ReduceScatterOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], ReduceScatter set inter "\
                "server nhr algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
    }
    char *pathvar;
    pathvar = getenv("ALG");

    if(pathvar != NULL){
        //设置后 打平算法
        algName = "AlignedReduceScatterAsymDoubleRingExecutor";
    }
    else
    {
        //默认分级算法
        algName = "AlignedReduceScatterAsymNewDoubleRingExecutor";
    }
    // algName = "AlignedReduceScatterAsymDoubleRingExecutor";
    // algName = "AlignedReduceScatterDoubleRingFor91093Executor";
    HCCL_INFO("[SelectAlgfor91093] reduce_scatter SelectAlgfor91093 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatter, ReduceScatterOperator);

}