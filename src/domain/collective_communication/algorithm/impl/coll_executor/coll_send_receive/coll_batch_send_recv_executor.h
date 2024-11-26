/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_SEND_RECV_EXECUTOR_H
#define COLL_BATCH_SEND_RECV_EXECUTOR_H

#include "coll_comm_executor.h"

namespace hccl {
class CollBatchSendRecvExecutor : public CollCommExecutor {
public:
    CollBatchSendRecvExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollBatchSendRecvExecutor() = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    // 增量建链资源计算接口
    HcclResult CalcIncreLinkRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;
protected:
    /* *************** 资源计算 *************** */
    void ParseParam(const OpParam& param) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;

    /* *************** 算法编排 *************** */
    HcclResult ProcessSelfSendRecvTasks(std::vector<HcclSendRecvItem*> &orderedList,
        u32 itemNum, u32& loopStartIndex, Stream& stream);
    u64 CalcSendLoopMaxCount(DeviceMem& inCCLBuffer, const u32 unitSize);
    u64 CalcRecvLoopMaxCount(DeviceMem& outCCLBuffer, const u32 unitSize);
    HcclResult GetSendRecvInfo(HcclSendRecvItem* itemPtr);

    u32 remoteUserRank_ = 0;
    HcclSendRecvType sendRecvType_;
private:
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult GetPairWiseList(HcclSendRecvItem *sendRecvItemsPtr, u32 itemNum,
        std::vector<HcclSendRecvItem *> &orderedList);
    HcclResult RunLoop(OpParam &param, AlgResourceResponse &algRes, HcclSendRecvItem* sendRecvItem);
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    std::set<u32> commTargetUserRankSet_;
};
} // namespace hccl

#endif