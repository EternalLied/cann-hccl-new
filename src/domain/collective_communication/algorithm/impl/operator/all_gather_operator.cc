/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_operator.h"
#include "device_capacity.h"
#include "rank_consistent.h"
#include "executor_impl.h"
#include "coll_alg_op_registry.h"

namespace hccl {
AllGatherOperator::AllGatherOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER)
{
}

AllGatherOperator::~AllGatherOperator()
{
}

HcclResult AllGatherOperator::SelectAlg(const std::string& tag, const OpParam& param, std::string& algName,
                                        std::string& newTag)
{
    if (userRankSize_ == 1 && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        algName = "AllGatherSingleExecutor";
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
        HCCL_ERROR("[AllGatherSelector][SelectAlg]tag[%s], all_gather failed, return[%d]", tag.c_str(), ret), ret);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        newTag = tag;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        newTag = tag + algName;
    } else {
        AlgTypeLevel1 algType1 = GetLevel1AlgType(algType_);
        auto level1Iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType1);
        CHK_PRT_RET(level1Iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(), HCCL_ERROR("level1: algType1[%u] is invalid.",
            algType1), HCCL_E_INTERNAL);
        newTag = tag + level1Iter->second + algName;
    }
    newTag += (param.aicpuUnfoldMode ? "_device" : "_host");
    HCCL_INFO("[SelectAlg] all_gather newTag is [%s]", newTag.c_str());
    return ret;
}

HcclResult AllGatherOperator::SelectAlgfor310P3(const OpParam& param, std::string& algName)
{
    algName = "AllGatherFor310PExecutor";
    HCCL_INFO("[SelectAlgfor310P3] all_gather SelectAlgfor310P3 is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910A(const OpParam& param, std::string& algName)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_4P_MESH || topoType_ == TopoType::TOPO_TYPE_2P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING || topoType_ == TopoType::TOPO_TYPE_8P_RING;

    if (isMeshTopo) {
        algName = "AllGatherMeshExecutor";
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910A] all_gather SelectAlgfor910A is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor910B(const OpParam& param, std::string& algName)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = param.DataDes.count * unitSize; // 单位：字节
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
        topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;
    bool isRingTopo = topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING;

    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isSingleMeshAggregation_) {
        u64 cclBufferSize = cclBufferManager_.GetOutCCLbufferSize() / userRankSize_;
        std::string algTypeLevel1Tag;
        CHK_RET(AutoSelectAlgTypeLevel1(HcclCMDType::HCCL_CMD_ALLGATHER, dataSize, cclBufferSize, algTypeLevel1Tag));
        if (param.opBaseAtraceInfo != nullptr) {
            CHK_RET(param.opBaseAtraceInfo->SavealgtypeTraceInfo(algTypeLevel1Tag, param.tag));
        }
    }

    // pipeline算法task数量多，如果超出FFTS子图限制，则重定向到HD算法
    if (GetLevel1AlgType(algType_) == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        u32 contextNum = CalcContextNumForPipeline(HcclCMDType::HCCL_CMD_ALLGATHER);
        if (contextNum > HCCL_FFTS_CAPACITY) {
            CHK_RET(SetInterServerHDAlgo(algType_));
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor910B] context num[%u] is out of capacityof FFTS+ graph[%u],"
                "reset algorithm to HD.", contextNum, HCCL_FFTS_CAPACITY);
        }
    }

    if (isMeshTopo) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            if (isSingleMeshAggregation_) {
                algName = "AllGatherMeshOpbaseExecutor";
            } else if (UseInterServerPipelineAlgo(algType_)) {
                algName = "AllGatherMeshOpbasePipelineExecutor";
            }
        }
        if (algName.empty()) {
            algName = "AllGatherMeshExecutor";
        }
    } else if (isRingTopo) {
        algName = "AllGatherRingExecutor";
    } else {
        algName = "AllGatherComm";
    }
    HCCL_INFO("[SelectAlgfor910B] all_gather SelectAlgfor910B is algName [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult AllGatherOperator::SelectAlgfor91093(const OpParam& param, std::string& algName)
{
    bool smallCountOptim91093 = (serverNum_ == 1) &&
        ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) ||
        (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !param.aicpuUnfoldMode)) &&
        (param.DataDes.count * SIZE_TABLE[param.DataDes.dataType] <= HCCL_SMALL_COUNT_2_MB) &&
        (deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO);
    if (multiModuleDiffDeviceNumMode_ || multiSuperPodDiffServerNumMode_) {
        algName = "AllGatherComm";
    } else if (smallCountOptim91093) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            algName = "AllGatherMeshOpbaseExecutor";
        } else {
            algName = "AllGatherMeshExecutor";
        }
    } else if (GetExternalInputEnableRdmaSdmaConcurrent() && topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !param.aicpuUnfoldMode && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)) {
        if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
            HcclResult ret = SetInterServerRingAlgo(algType_);
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] concurrent only support ring or NB in AlgoLevel1 "\
                "yet, default is ring.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], AllGather concurrent "\
                    "set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        algName = "AllGatherDoubleRingConcurrentExecutor";
        // algName = "AllGatherDoubleRingAsymExecutor";
    } else {
        if (GetExternalInputEnableRdmaSdmaConcurrent()) {
            if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_))) {
                HcclResult ret = SetInterServerRingAlgo(algType_);
                CHK_PRT_RET(ret != HCCL_SUCCESS,
                    HCCL_ERROR("[AllGatherOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], AllGather concurrent "\
                    "set inter server ring algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
            }
        } else if (!(UseInterServerRingAlgo(algType_) || UseInterServerNBAlgo(algType_) ||
            UseWholeRingAlgo(algType_))) {
            HcclResult ret = SetInterServerNHRAlgo(algType_);
            HCCL_WARNING("[AllGatherOperator][SelectAlgfor91093] only support ring, NB and NHR in AlgoLevel1 yet, "\
                "default is algType=NHR.");
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[AllGatherOperator][SelectAlgfor91093]errNo[0x%016llx] tag[%s], AllGather set inter server "\
                    "nhr algo failed", HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);
        }
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            // algName = "AlignedAllGatherDoubleRingFor91093Executor";
            algName = "AlignedAllGatherAsymDoubleRingExecutor";
        } else if (topoType_ == TopoType::TOPO_TYPE_NP_SINGLE_RING){
            algName = "AllGatherRingFor91093Executor";
        } else {
            algName = "AllGatherComm";
        }
    }

    char *pathvar;
    pathvar = getenv("ALG");

    if(pathvar != NULL){
        //设置后 打平算法
        algName = "AlignedAllGatherAsymDoubleRingExecutor";
    }
    else
    {
        //默认分级算法
        algName = "AlignedAllGatherAsymNewDoubleRingExecutor";
    }
    // algName = "AlignedAllGatherAsymDoubleRingExecutor";
    // HCCL_INFO("[SelectAlgfor91093] all_gather SelectAlgfor91093 is algName [%s]", algName.c_str());
    // algName = "AllGatherDoubleRingAsymExecutor";
    // algName = "AllGatherDoubleRingConcurrentExecutor";
    // algName = "AlignedAllGatherAsymDoubleRingExecutor";
    // algName = "AlignedAllGatherDoubleRingFor91093Executor";
    HCCL_INFO("[SelectAlgfor91093] all_gather SelectAlgfor91093 is algName [%s] finally", algName.c_str());
    return HCCL_SUCCESS;
}

REGISTER_OP(HcclCMDType::HCCL_CMD_ALLGATHER, AllGather, AllGatherOperator);

}