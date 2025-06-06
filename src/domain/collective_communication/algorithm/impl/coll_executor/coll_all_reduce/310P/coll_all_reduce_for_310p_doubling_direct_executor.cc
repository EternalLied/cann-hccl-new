/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_for_310p_doubling_direct_executor.h"

namespace hccl {
CollAllReduceFor310PDoublingDirectExecutor::CollAllReduceFor310PDoublingDirectExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher): CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
}

HcclResult CollAllReduceFor310PDoublingDirectExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingDirectExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;
    HCCL_INFO("[CollAllReduceFor310PDoublingDirectExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingDirectExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollAllReduceFor310PDoublingDirectExecutor][CalcOuterCommInfo]tag[%s] start", tag_.c_str());

    if (algType_ == AlgType::ALG_NP_HD) {
        CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_HALVING_DOUBLING);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    } else if (algType_ == AlgType::ALG_DEFAULT) {
        CommParaInfo commParaInfo(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaInfo, opTransport[COMM_LEVEL0], inputType, outputType));
    } else {
        HCCL_ERROR("unsupported algType %d", algType_);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[CollAllReduceFor310PDoublingDirectExecutor][CalcOuterCommInfo]tag[%s] Calc RingComm finish",
        tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceFor310PDoublingDirectExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    bool isInlineReduce = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.outputMem.ptr(),
        param.DataDes.dataType, param.reduceType);
    u64 reduceAttr = 0;
    if (isInlineReduce) {
        SalSetBitOne(reduceAttr, ATTR_POS_INLINE_REDUCE);
    }

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType
    };

    std::unique_ptr<ExecutorBase> executor;
    executor.reset(new (std::nothrow) AllReduceDoublingDirect(dispatcher_, reduceAttr, &opInfo));
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count,
        param.DataDes.dataType, param.stream, param.reduceType,
        OUTER_BRIDGE_RANK_ID, std::vector<Slice>(0), 0));

    CHK_RET(executor->RegisterProfiler(
        (outerCommInfo.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + outerCommInfo.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(executor, outerCommInfo));
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceDoublingDirect", AllReduceFor310PDoublingDirect, CollAllReduceFor310PDoublingDirectExecutor);
} // namespace hccl