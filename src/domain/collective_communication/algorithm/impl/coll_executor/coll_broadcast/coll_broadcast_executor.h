/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BROADCAST_COMM_EXECUTOR_H
#define COLL_BROADCAST_COMM_EXECUTOR_H
#include "coll_comm_executor.h"
#include "coll_alg_operator.h"

namespace hccl {
class CollBroadcastExecutor : public CollCommExecutor {

public:
    CollBroadcastExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBroadcastExecutor() = default;

    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
protected:
    /* *************** 算法编排 *************** */
    // Broadcast Loop Executor公共接口
    HcclResult RunLoop(OpParam &param, AlgResourceResponse &algRes);
    HcclResult GetSliceNum(const u64 totalSize, const bool isSmallData, u64& sliceNum);
    bool IsBroadcastSmallData(u64 size, u64 totalSize);
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);
    virtual u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
    HcclResult GetRankSliceSize(HcclDataType dataType, const u64 count, const u32 rankSize,
                std::vector<Slice> &sliceList);
    bool DMAReduceFlag_{false}; // 是否DMA消减
    bool scratchMemFlag_{false};  // 是否需要申请scratch memory，不需要申请则传入outputmem为scratchmem

private:
    HcclResult RunLoopInner(OpParam &param, ExecMem &execMem);
};
} // namespace hccl

#endif