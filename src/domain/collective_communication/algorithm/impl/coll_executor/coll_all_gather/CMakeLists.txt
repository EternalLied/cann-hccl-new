set(src_list
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_comm_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_ring_for_910_93_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_opbase_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_mesh_opbase_pipeline_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_single_rank_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_double_ring_concurrent_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_all_gather_double_ring_for_910_93_executor.cc

  ${CMAKE_CURRENT_SOURCE_DIR}/coll_all_gather_double_ring_asym_executor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/coll_aligned_all_gather_asym_double_ring_executor.cc
)

target_sources(hccl_alg PRIVATE
  ${src_list}
)

add_subdirectory(310P)