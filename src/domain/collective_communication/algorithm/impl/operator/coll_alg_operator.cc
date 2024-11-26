/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>

#include "coll_alg_operator.h"
#include "device_capacity.h"
#include "nonuniform_hierarchical_ring_base_pub.h"
#include "coll_executor_base.h"
#include "coll_alg_exec_registry.h"


namespace hccl {
using namespace std;
constexpr float GB2B = 1024 * 1024 * 1024;
constexpr float SECOND2MICROSECOND = 1000000;
constexpr float RHD_FACTOR_TWO = 2.0;
constexpr float RHD_FACTOR_ONE = 1.0;
constexpr float DOUBLE_SUB_HCCLCMD = 2.0; // The hcclCMD can be considered as combination of two hcclCMDs.
constexpr float COPY_TIME_IN_RHD = 1.0;
constexpr double NHR_FACTOR_TWO = 2.0;
constexpr double NHR_FACTOR_THREE = 3.0;
constexpr double NHR_FACTOR_FOUR = 4.0;
constexpr double NHR_SUB_TWO = 2.0;
constexpr float LATENCY = 60; // 静态时延 60 us;
constexpr u64 PIPELINE_MIN_SIZE = 32 * 1024; // 当数据量大于等于32KB时，reduce_scatter和all_gather使能pipeline模式
constexpr u64 PIPELINE_ALLREDUCE_MIN_SIZE = 1024 * 1024; // 当数据量大于等于1MB时，allreduce使能pipeline模式
constexpr u64 PIPELINE_MIN_SIZE_NO_LITE = 2 * 1024 * 1024; // 如不支持RDMALite，当数据量大于等于2MB时，使能pipeline模式
constexpr u64 HCCL_FFTS_CAPACITY = 65535; // FFTS+子图最大容量

CollAlgOperator::CollAlgOperator(AlgConfigurator* algConfigurator, CCLBufferManager &cclBufferManager,
                                 HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher,
                                 HcclCMDType opType)
    : algConfigurator_(algConfigurator), cclBufferManager_(cclBufferManager),
      dispatcher_(dispatcher), topoMatcher_(topoMatcher),
      workflowMode_(GetWorkflowMode())
{
    SetTopoAttr(algConfigurator_);
    SetAlgoAttr(algConfigurator_);
    algConfigurator->GetAlgTypeDirect(algType_, opType);
    algConfigurator->GetAlgoLevel1DefaultSwitch(isAlgoLevel1Default_, opType);
    algConfigurator->GetTopoType(topoType_);
}

HcclResult CollAlgOperator::SelectAlg(const std::string& tag,
    const OpParam& param, std::string& algName, std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::CalcResRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }
    return executor_->CalcResRequest(param, resourceRequest);
}

HcclResult CollAlgOperator::Orchestrate(const std::string& algName, OpParam& param, AlgResourceResponse& algResource)
{
    HCCL_INFO("[CollAlgOperator][Orchestrate]algName[%s]", algName.c_str());
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[CollAlgOperator][Orchestrate]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(SetExecutorAttr(param));
    }

    return executor_->Orchestrate(param, algResource);
}

HcclResult CollAlgOperator::CalcIncreLinkRequest(const std::string& algName, const OpParam& param,
    AlgResourceRequest& resourceRequest)
{
    if (executor_.get() == nullptr) {
        executor_ = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(executor_.get() == nullptr,
            HCCL_ERROR("[BatchSendRecvOperator][CalcIncreLinkRequest]Fail to find executor for algName[%s]",
            algName.c_str()), HCCL_E_PARA);
    }
    return executor_->CalcIncreLinkRequest(param, resourceRequest);
}

void CollAlgOperator::SetTopoAttr(AlgConfigurator* algConfigurator)
{
    const HcclTopoAttr& topoAttr = algConfigurator->GetTopoAttr();
    serverNum_= topoAttr.serverNum;
    moduleNum_ = topoAttr.moduleNum;
    superPodNum_ = topoAttr.superPodNum;
    deviceNumPerServer_ = topoAttr.deviceNumPerServer;
    deviceNumPerAggregation_ = topoAttr.deviceNumPerAggregation;
    multiModuleDiffDeviceNumMode_ = topoAttr.multiModuleDiffDeviceNumMode;
    multiSuperPodDiffServerNumMode_ = topoAttr.multiSuperPodDiffServerNumMode;

    meshAggregationRankSize_ = topoAttr.meshAggregationRankSize;
    isDiffDeviceModule_ = topoAttr.isDiffDeviceModule;
    isSingleMeshAggregation_ = topoAttr.isSingleMeshAggregation;
    isAllRankSamePlane_ = topoAttr.isAllRankSamePlane;
    is310PDuoCard_ = topoAttr.is310PDuoCard;

    userRank_ = topoAttr.userRank;
    realUserRank_ = topoAttr.realUserRank;
    userRankSize_ = topoAttr.userRankSize;

    devicePhyId_ = topoAttr.devicePhyId;
    deviceLogicId_ = topoAttr.deviceLogicId;
    deviceType_ = topoAttr.deviceType;

    nicList_ = topoAttr.nicList;
    pairLinkCounter_ = topoAttr.pairLinkCounter;
    isSupportRdmaLite_ = topoAttr.isSupportRdmaLite;
    useSuperPodMode_ = topoAttr.useSuperPodMode;
    return;
}

void CollAlgOperator::SetAlgoAttr(AlgConfigurator* algConfigurator)
{
    const HcclAlgoAttr& algoAttr = algConfigurator->GetAlgoAttr();
    isHaveCpuRank_ = algoAttr.isHaveCpuRank;
    inlineReduceSwitchOn_ = algoAttr.inlineReduceSwitchOn;
    identifier_ = algoAttr.identifier;
    return;
}

HcclResult CollAlgOperator::SetExecutorAttr(const OpParam& param)
{
    CHK_RET(executor_->SetAlgType(algType_));
    CHK_RET(executor_->SetCCLInBuffer(cclBufferManager_.GetInCCLbufferSize()));

    if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        bool isSupportSDMAReduce = false;
        if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            isSupportSDMAReduce = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType,
                param.reduceType);
        } else {
            isSupportSDMAReduce = IsSupportSDMAReduce(cclBufferManager_.GetInCCLbuffer().ptr(),
                cclBufferManager_.GetOutCCLbuffer().ptr(), param.DataDes.dataType, param.reduceType);
        }
        CHK_RET(executor_->SetIsSupportSDMAReduce(isSupportSDMAReduce));
    }
    return HCCL_SUCCESS;
}

std::string CollAlgOperator::GenerateNewTagByAlgTypeLevel1(std::string tag, std::string algTypeLevel1Tag) const
{
    if (algTypeLevel1Tag == "") {
        return tag;
    } else {
        return tag + "_" + algTypeLevel1Tag;
    }
}

HcclResult CollAlgOperator::AppendTag(const AlgTypeLevel1 &algTypeLevel1, std::string &tag)
{
    switch (algTypeLevel1) {
        case AlgTypeLevel1::ALG_LEVEL1_RING:
            tag = "ALG_LEVEL1_RING";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_HD:
            tag = "ALG_LEVEL1_HD";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_NHR:
            tag = "ALG_LEVEL1_NHR";
            break;
        case AlgTypeLevel1::ALG_LEVEL1_PIPELINE:
            tag = "ALG_LEVEL1_PIPELINE";
            break;
        default:
            HCCL_WARNING("[CollAlgOperator][AppendTag] The algTypeLevel1 %d is not supported.", algTypeLevel1);
            break;
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::AutoSelectAlgTypeLevel1(HcclCMDType hcclCMDType, u64 countSize, u64 cclBufferSize,
                                                    std::string &algTypeLevel1Tag, bool isInlineReduce,
                                                    bool isRdmaReduce, bool isAivMode)
{
    if (isSingleMeshAggregation_) {
        HCCL_INFO("[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, no need to choose algo.",
                  serverNum_, moduleNum_);
        return HCCL_SUCCESS;
    }

    // auto algo selection process
    if (isAlgoLevel1Default_) {
        // parse algType_ and get algTypeLevel1 and algTypeLevel0
        auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
        // set algTypeLevel1
        AlgTypeLevel1 algTypeLevel1;
        CHK_RET(
            GetDefaultAlgoLevel1V2(
                hcclCMDType, countSize, cclBufferSize, algTypeLevel1, isInlineReduce, isRdmaReduce, isAivMode));
        auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algTypeLevel1);
        CHK_PRT_RET(iter == HCCL_ALGO_LEVEL1_NAME_MAP.end(),
            HCCL_ERROR("[AutoSelectAlgTypeLevel1] level1: algType[%u] is invalid.", algTypeLevel1),
            HCCL_E_INTERNAL);
        HCCL_INFO("[AutoSelectAlgTypeLevel1] there are %u server(%u module) in level1, using %s algo",
                  serverNum_, moduleNum_, iter->second.c_str());
        algType_ = AlgType(
            (static_cast<u32>(algTypeLevel1) << HCCL_LEVEL_ALGO_WIDTH) + static_cast<u32>(originalAlgTypeLevel0));
        // tag 增加所选的算法
        AppendTag(algTypeLevel1, algTypeLevel1Tag);
    }
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoForComm(HcclCMDType hcclCMDType, float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    // 从map中查找对应的计算函数
    auto it = selectFuncMap_.find(hcclCMDType);
    if (it == selectFuncMap_.end()) {
        HCCL_ERROR("[Get][AlgTypeLevel1] The hcclCMDType %d is not supported.", hcclCMDType);
        return HCCL_E_NOT_SUPPORT;
    }
    return (it->second)(delay, curSize, bandWidth, algType);
}

// 保守估计Pipeline算法所需context数量
u32 CollAlgOperator::CalcContextNumForPipeline(HcclCMDType hcclCMDType)
{
    const u32 stepNum = moduleNum_;  // 通信步数
    const u32 hccsContextNumPerStep = 5 * (deviceNumPerAggregation_ - 1);   // SDMA跨片每步所需context数
    const u32 roceContextNumPerStep = 7;  // RDMA每步所需context数
    const u32 copyContextNumPerStep = 1;  // SDMA片内每步所需context数
    const u32 contextNumPerStep = hccsContextNumPerStep + roceContextNumPerStep + copyContextNumPerStep; // 小计
    const u32 barrierContextNum = 4;  // 通信结束时barrier操作所需context数

    switch (hcclCMDType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:             // fall-through
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:        // fall-through
        case HcclCMDType::HCCL_CMD_ALLGATHER: {
            const u32 copyContextNum = 1;    // 通信首尾所需context数量
            u32 contextNum = stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
            if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
                contextNum += contextNum;
            }
            return contextNum;
        }
        case HcclCMDType::HCCL_CMD_ALLTOALLV:             // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:            // fall-through
        case HcclCMDType::HCCL_CMD_ALLTOALL: {
            const u32 copyContextNum = 1 + moduleNum_;   // 通信首尾所需context数量
            return stepNum * contextNumPerStep + barrierContextNum + copyContextNum;
        }
        default:
            return 0;
    }
}

HcclResult CollAlgOperator::GetDefaultAlgoLevel1V2(HcclCMDType hcclCMDType, u64 curSize, u64 cclBufferSize,
    AlgTypeLevel1 &algType, bool isInlineReduce, bool isRdmaReduce, bool isAivMode)
{
    // pipeline mode is deployed,where there is multi-sever multi-device(insever) now,
    // since RDMA is not reduced by normal serial orchestration of tasks.
    // So pipeline mode is more dominant than normal serial orchestration now.
    auto originalAlgTypeLevel0 = GetLevel0AlgType(algType_);
    bool disdeterniminsticWithInlineReduce = isInlineReduce && isRdmaReduce &&
        topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE;

    // 对于不支持Rdma Lite的场景，下发性能较差，RS和AG需要一个很大的数据量（AR的一半）才能掩盖下发时间
    u64 pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (((hcclCMDType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && disdeterniminsticWithInlineReduce) ||
        hcclCMDType == HcclCMDType::HCCL_CMD_ALLGATHER) &&
        deviceNumPerAggregation_ != 1 && curSize >= pipelineMinSize && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
        CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
        algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
        return HCCL_SUCCESS;
    }

    // 对于不支持Rdma Lite的场景，下发性能较差，AllReduce需要一个较大的数据量才能掩盖下发时间
    pipelineMinSize = (isSupportRdmaLite_) ? (PIPELINE_ALLREDUCE_MIN_SIZE) : (PIPELINE_MIN_SIZE_NO_LITE);
    if (hcclCMDType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // 计算每个slice的大小
        u64 allreduceCurSize = 0;
        allreduceCurSize = curSize / (moduleNum_ * deviceNumPerAggregation_);
        if (disdeterniminsticWithInlineReduce && deviceNumPerAggregation_ != 1 &&
            allreduceCurSize >= pipelineMinSize && !isAivMode && IsAlgTypeLevel0Mesh(originalAlgTypeLevel0) &&
            CalcContextNumForPipeline(hcclCMDType) <= HCCL_FFTS_CAPACITY) {
            algType = AlgTypeLevel1::ALG_LEVEL1_PIPELINE;
            return HCCL_SUCCESS;
        }
    }
    u64 dataSizePerLoop = curSize > cclBufferSize ? cclBufferSize : curSize;
    float delay = LATENCY; // 静态时延 60 us;
    float bandWidth;
    CHK_RET(GetBandWidthPerNPU(1, userRankSize_, deviceNumPerAggregation_, bandWidth)); // 单位：GB/s
    bandWidth = bandWidth * GB2B; // 单位：B/s
    CHK_RET(SelectAlgoForComm(hcclCMDType, delay, dataSizePerLoop, bandWidth, algType));
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduceScatter(float delay, u64 recvCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                recvCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD,
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 recvCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = ceil(log2(moduleNum_)) * delay +
                static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                sendCurSize * userRankSize_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForGather(float delay, u64 sendCurSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = static_cast<double>(steps) * delay +
                      static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = log2(moduleNum_) * delay +
                 static_cast<double>(steps) / moduleNum_ * sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = ceil(log2(moduleNum_)) * delay +
                 static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 sendCurSize * userRankSize_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForAllReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;

    // theoretical time cost of NHR
    double nhrCost = NHR_FACTOR_TWO * ceil(log2(moduleNum_)) * delay +
                NHR_FACTOR_TWO * static_cast<double>(moduleNum_ - 1) / moduleNum_ *
                curSize / deviceNumPerAggregation_ / bandWidth * SECOND2MICROSECOND;

    // compare costs bewteen NHR and Ring, if same cost, Ring > NHR > HD
    algType = (nhrCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_NHR : AlgTypeLevel1::ALG_LEVEL1_RING;
    double interMinCost = min(nhrCost, ringCost);

    // theoretical time cost of HD/RHD
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // the (RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ is equal to 1 + (moduleNum_ -1) / moduleNum_
        hdCost = DOUBLE_SUB_HCCLCMD * ceil(log2(moduleNum_)) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(RHD_FACTOR_TWO * moduleNum_ - RHD_FACTOR_ONE) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }

    // compare cost among NHR, HD and Ring
    algType = (hdCost < interMinCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : algType;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForBroadcast(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth
                 * SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = scatter + allgather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

HcclResult CollAlgOperator::SelectAlgoTypeForReduce(float delay, u64 curSize, float bandWidth,
    AlgTypeLevel1 &algType)
{
    auto steps = moduleNum_ - 1;
    // theoretical time cost of Ring
    double ringCost = DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) * delay +
                      DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                      curSize / deviceNumPerAggregation_ / bandWidth *
                      SECOND2MICROSECOND;
    double hdCost = 0.0;
    if ((moduleNum_ & (moduleNum_ - 1)) == 0) {
        // theoretical time cost of HD
        hdCost = DOUBLE_SUB_HCCLCMD * log2(moduleNum_) * delay +
                 DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_ *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    } else {
        // theoretical time cost of Recursive HD
        // rhd-broadcast = reducescatter + gather + copy
        hdCost = (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * floor(log2(moduleNum_))) * delay +
                 (COPY_TIME_IN_RHD + DOUBLE_SUB_HCCLCMD * static_cast<double>(steps) / moduleNum_) *
                 curSize / deviceNumPerAggregation_ / bandWidth *
                 SECOND2MICROSECOND;
    }
    algType = (hdCost < ringCost) ? AlgTypeLevel1::ALG_LEVEL1_HD : AlgTypeLevel1::ALG_LEVEL1_RING;
    return HCCL_SUCCESS;
}

AlgType CollAlgOperator::GetAlgType()
{
    return algType_;
}

bool CollAlgOperator::Is2U2PInfer()
{
    return ((deviceNumPerAggregation_ == HCCL_DEVICE_NUM_TWO) && (serverNum_ == 1) &&
            (deviceType_ == DevType::DEV_TYPE_910B) && (meshAggregationRankSize_ == HCCL_DEVICE_NUM_TWO) &&
            (pairLinkCounter_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)] == 0));
}

bool CollAlgOperator::Is910BSingleMesh()
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isSingleMesh =
        (deviceType_ == DevType::DEV_TYPE_910B) && (isMeshTopo || Is2U2PInfer()) && (userRankSize_ != 1);
    return isSingleMesh;
}

bool CollAlgOperator::NeedCreateSingleMeshPlane(const bool isInlineReduce)
{
    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool meshSinglePlane = Is910BSingleMesh() && topoMatcher_->GetDeterministicConfig() == DETERMINISTIC_CONFIG_DISABLE &&
        isInlineReduce && (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    return meshSinglePlane;
}

bool CollAlgOperator::SingleMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool singleMeshInlineReduce = Is910BSingleMesh() && isInlineReduce && isSingleMeshAggregation_;
    return singleMeshInlineReduce;
}

bool CollAlgOperator::IsMultiMeshInlineReduce(void *inputPtr, void *outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    bool isMeshTopo = topoType_ == TopoType::TOPO_TYPE_NP_MESH || topoType_ == TopoType::TOPO_TYPE_4P_MESH ||
                      topoType_ == TopoType::TOPO_TYPE_2P_MESH || topoType_ == TopoType::TOPO_TYPE_1P_MESH;

    bool isInlineReduce = IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op);
    bool isRdmaReduce = IsSupportRDMAReduce(dataType, op);
    bool multiMeshInlineReduce = (deviceType_ == DevType::DEV_TYPE_910B) &&
                                 isMeshTopo && isInlineReduce && isRdmaReduce && (!isSingleMeshAggregation_);
    return multiMeshInlineReduce;
}

void CollAlgOperator::SetLegacyHcclImpl(std::unique_ptr<hcclImpl> &impl)
{
    hcclImpl_ = impl.get();
    return;
}

bool CollAlgOperator::CheckUserInMemNotLargerThanCCLInMem(const HcclCMDType &opType, OpParam &param)
{
    u32 unitSize = SIZE_TABLE[param.DataDes.dataType];
    u64 dataSize = 0;
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        dataSize = param.DataDes.count * unitSize * userRankSize_;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        dataSize = param.DataDes.count * unitSize;
    }

    void *commInputPtr = nullptr;
    u64 commInputSize = 0;
    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    if (dataSize <= commInputSize) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] <= CCLInMem[%llu]", dataSize, commInputSize);
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU] UserInMem[%llu] > CCLInMem[%llu]", dataSize, commInputSize);
    }
    return dataSize <= commInputSize;
}

bool CollAlgOperator::ExecutorOnlySupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceMeshSmallCountExecutor") || (algName == "ReduceScatterDeterExecutor");
}

bool CollAlgOperator::ExecutorCanSupportDMAReduce(const std::string& algName)
{
    const std::set<std::string> executorCanSupportDMAReduceSet = {
        "AllReduceRingFor91093Executor", "AllReduceDoubleRingConcurrentExecutor",
        "AllReduceFastDoubleRingFor91093Executor", "AlignedAllReduceDoubleRingFor91093Executor",
        "ReduceScatterRingFor91093Executor", "ReduceScatterDoubleRingConcurrentExecutor",
        "ReduceScatterFastDoubleRingFor91093Executor", "AlignedReduceScatterDoubleRingFor91093Executor"
        };
    if (executorCanSupportDMAReduceSet.find(algName) != executorCanSupportDMAReduceSet.end()) {
        return true;
    }
    return false;
}

bool CollAlgOperator::ExecutorNoSupportDMAReduce(const std::string& algName)
{
    return (algName == "AllReduceComm") || (algName == "ReduceScatterComm");
}

bool CollAlgOperator::ExecutorSupportInPlace(OpParam &param, std::string& algName, u8 &inPlaceSupportRetryStatus)
{
    // case 2.2
    if (ExecutorOnlySupportDMAReduce(algName)) {
        if (param.retryEnable) {
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_ONE;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorOnlySupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_TWO;
        return false;
    } else if (ExecutorNoSupportDMAReduce(algName)) {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorNoSupportDMAReduce[%s] is allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_THREE;
        return true;
    } else if (ExecutorCanSupportDMAReduce(algName)) {
        if (param.retryEnable) {
            // 对应的executor会感应RetryEnable环境变量，走非DMA削减逻辑
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
                " for inplace case, the executor without DMAReduce will be applied.", algName.c_str());
            inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_FOUR;
            return true;
        }
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]ExecutorCanSupportDMAReduce[%s] is not allowed"
            " for inplace case.", algName.c_str());
        inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_FIVE;
        return false;
    } else {
        HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The unknown executor[%s] does not support "
            "for an inplace case yet.", algName.c_str());
        inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_SIX;
        return false;
    }
}

bool CollAlgOperator::FitRetryConditionforInPlaceOp(
    const HcclCMDType &opType, OpParam &param, std::string& algName, u8 &inPlaceSupportRetryStatus)
{
    // case 1 allgather or broadcast
    if (opType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_ZERO;
        return true;
    }
    // case 2 reducescatter or allreduce
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER ||
        opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        // case 2.1
        if (CheckUserInMemNotLargerThanCCLInMem(opType, param)) {
            // case 2.4: 在hccl_communicator.cc的ExecOp之前已经该让图模式走单算子模式了，理论上不会进入此条件
            HCCL_INFO("[CollAlgOperator][OpRetry][AICPU]The retry with inplace case is expected to be supported, "
                "therefore HcclWorkflowMode is set to [%d]",
                static_cast<u8>(GetWorkflowMode()));
            return ExecutorSupportInPlace(param, algName, inPlaceSupportRetryStatus);
        } else {
            // case 2.3 UsrIn > CCLIn
            inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_SEVEN;
            return false;
        }
    }
    // 其他算子类型不支持
    inPlaceSupportRetryStatus = INPLACE_SUPPORT_RETRY_STATUS_EIGHT;
    return false;
}

bool CollAlgOperator::SupportRetryWithInplaceCheck(
    const HcclCMDType &opType, OpParam &param, std::string& algName, u8 &isInplaceStatus,
    u8 &inPlaceSupportRetryStatus)
{
    // 不支持inplace的通信算子重执行
    if (IsHcclOpInplace(opType, param, userRank_, userRankSize_, isInplaceStatus)) {
        if(!FitRetryConditionforInPlaceOp(opType, param, algName, inPlaceSupportRetryStatus)) {
            HCCL_DEBUG("[CollAlgOperator][OpRetry][AICPU]hccl aicpu can not retry, opType[%s], inputPtr[0x%016lx], "
                "outputPtr[0x%016lx].",
                GetCMDTypeEnumStr(opType).c_str(), param.inputPtr, param.outputPtr);
            return false;
        }
    }
    // true 存在两种情况：
    // 1. 非inplace场景
    // 2. 是inplace但同时符合retry条件的场景
    return true;
}
}   // namesapce hccl
