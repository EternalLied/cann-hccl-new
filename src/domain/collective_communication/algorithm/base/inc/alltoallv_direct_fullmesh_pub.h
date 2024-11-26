/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_DIRECT_FULLMESH_PUB_H
#define ALLTOALL_V_DIRECT_FULLMESH_PUB_H

#include "alltoallv_staged_calculator_pub.h"

namespace hccl {
const uint32_t ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE =  16; // SDMA链路上的并发数量
const uint32_t ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE =  1; // RDMA链路上的并发数量

class AlltoAllVDirectFullMesh : public ExecutorBase {
public:
    explicit AlltoAllVDirectFullMesh(const HcclDispatcher dispatcher, Stream &mainStream,
        u32 userRank, u32 userRankSize, const std::vector<LINK> &links,
        const SendRecvInfo &tmpRankSendRecvInfo, u32 podNum, u32 devNumInlocalPod,
        u32 rankIdxInPod);
    ~AlltoAllVDirectFullMesh() override;
    HcclResult Prepare(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &cclInMem,
        DeviceMem &cclOutMem, HcclWorkflowMode workMode,
        std::vector<Stream> &subStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain);
    HcclResult RunAsync();

protected:
private:
    HcclResult GenerateSubStreamInfo(std::vector<Stream> &subStreams,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain);
    std::string GetStreamIndexString();
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    u32 CalcNumSubStep();
    HcclResult NotifyRemoteRankStart(u32 step);
    HcclResult SDMAwithRemoteRankAndNotifyEnd(u32 step);
    HcclResult SendRecvData(u32 step);

    void UpdateCurrRankSendInfo(u32 destRank, std::vector<SendDataBlock>& sendInfo, u32 maxSendStep);
    void UpdateCurrRankRecvInfo(u32 destRank, std::vector<ReadDataBlock>& readInfo, u32 maxRecvStep);
    void UpdateOpBaseSubStreamInfo();
    void UpdatePartialCommunicationRankSet(u64 roundIdx, u32 groupRankSize);
    HcclResult PrepareIntraData(u32 step);
    HcclResult LocalCopy();
    HcclResult RunGroupFullMeshAlltoall(u32 step);
    HcclResult RunSDMA(HcclOpMetaInfoDef &opMeta);

    // RDMA处理相关函数
    HcclResult MainNotifyRdmaControlStart();
    HcclResult RdmaControlNotifyMainFinish();
    HcclResult RdmaControlNotifySubStart();
    HcclResult SubNotifyRdmaControlFinish();
    u32 GetNextDstRank(u32& curDstRank);
    u32 GetPreSrcRank(u32& curDstRank);
    void GenRdmaSendInfo(u32 dstRank, std::vector<SendDataBlock>& sendInfo);
    void GenRdmaRecvInfo(u32 srcRank, std::vector<RecvDataBlock>& recvInfo);
    HcclResult CopyDataForSend(u32 dstRank, std::vector<SendDataBlock>& sendInfo, u32 curStep, Stream strem);
    HcclResult SendRecvRdmaData(u32 dstRank, u32 srcRank, std::vector<SendDataBlock>& sendInfo,
        std::vector<RecvDataBlock>& recvInfo, u32 curStep, Stream strem);
    HcclResult CopyRecvDataToOutput(u32 srcRank, std::vector<RecvDataBlock>& recvInfo,
        u32 curStep, Stream strem);
    HcclResult ProcessSingleGroupRdmaData(std::vector<u32>& dstRanks, std::vector<u32>& srcRanks);
    HcclResult ProcessRdmaData();
    HcclResult RunRDMA();

    Stream mainStream_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalSubToMain_;
    u32 userRank_;
    u32 userRankSize_;
    u32 podStartRank_;  // 表示一个pod内起始的userRankId
    u32 podEndRank_; // 表示一个pod内结束的userRankId
    const std::vector<LINK> links_;
    const SendRecvInfo& localSendRecvInfo_;
    u32 podNum_;
    u32 devNumInlocalPod_;
    u32 rankIdxInPod_;
    u32 totalRdmaRankNum_; // 需要通信的rdma对端

    DeviceMem userInput_;
    DeviceMem userOutput_;
    DeviceMem cclInMem_;
    DeviceMem cclOutMem_;
    HcclWorkflowMode workMode_;
    u64 sdmaDataBlockSize_ = 0;

    bool islocalCpyDone_ = false;
    std::unordered_map<u32, std::vector<SendDataBlock>> subStreamSendInfo_; // 从流当前发送长度和发送的本地偏移
    std::unordered_map<u32, std::vector<ReadDataBlock>> subStreamReadInfo_; // 从流当前接收长度和接收到的本地偏移
    std::unordered_map<u32, u32> sendNumSubStep_;                       // 需要向对应对端rank发几次数据
    std::unordered_map<u32, u32> recvNumSubStep_;                       // 需要从对应对端rank收几次数据
    u32 sdmaConcurrentNum_; // 分组mesh-每组group的ranksize
    std::vector<std::pair<u32,u32>> partialCommRankSet_;  // 参与通信的rank组合

    // SDMA处理相关
    std::vector<Stream> sdmaSubStream_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalMainToSub_;
    std::vector<std::shared_ptr<LocalNotify>> sdmaMeshSignalSubToMain_;
    // RDMA处理相关
    u64 rdmaDataBlockSize_ = 0;
    // RDMA并发数量
    u32 rdmaConcurrentNum_;
    std::shared_ptr<LocalNotify> main2RdmaControlStreamNotify_;
    std::shared_ptr<LocalNotify> rdmaControl2MainStreamNotify_;
    // RDMA从流，以及RDMA控制流与从流同步的notify
    std::vector<Stream> rdmaSubStreams_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaControl2SubNotifies_;
    std::vector<std::shared_ptr<LocalNotify>> rdmaSub2ControlNotifies_;
};
} // namespace hccl
#endif /* ALLTOALL_V_MESH_READ_ONLY_PUB_H */