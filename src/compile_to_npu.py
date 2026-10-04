import struct
from dataclasses import dataclass
from typing import Optional, Tuple, List

# =============================================================================
# Shoot-and-Go ISA constants
# =============================================================================

RS2_FMT = "<12Q9I"
RS2_SIZE = struct.calcsize(RS2_FMT)  # 12*8 + 9*4 = 132 bytes
assert RS2_SIZE == 132

# hw_enables sub-fields / enum-like values
ALU_BYPASS = 0
ALU_ADD = 1
ALU_MUL = 2

INPUT_MXU = 0
INPUT_VPU1 = 1
INPUT_VPU2 = 2

OUT_VPU2 = 0
OUT_VPU1 = 1

NORM_BYPASS = 0
NORM_RMS = 1
NORM_LAYER = 2


def _u32(value: int, name: str) -> int:
    if not 0 <= int(value) <= 0xFFFF_FFFF:
        raise ValueError(f"{name} must fit uint32: {value}")
    return int(value)


def _u64(value: int, name: str) -> int:
    if not 0 <= int(value) <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"{name} must fit uint64: {value}")
    return int(value)


# =============================================================================
# [1] rs1: RoCC immediate control word
# =============================================================================

def build_rs1(
    # [15:0] vector scalar operand. 0 means unused.
    constant_operand: int = 0,
    # [31:16] output write-mask bitmap
    valid_row: int = 0xFFFF,
    # [32], [33]
    vector_compact_out: int = 0,
    vector_compact_in: int = 0,
    # [35:34], [37:36]
    out_point: int = OUT_VPU2,
    input_point: int = INPUT_MXU,
    # [38], [40:39]
    tile_strided_wr: int = 0,
    tile_strided_rd: int = 0,
    # [41], [43:42]
    transpose_en_wr: int = 0,
    transpose_en_rd: int = 0,
    # hw_enables [50:44]
    rope_en: int = 0,
    norm_mode: int = NORM_BYPASS,
    act_en: int = 0,
    alu_mode: int = ALU_BYPASS,
    mxu_en: int = 0,
    # [55:51], [56], [57]
    lut_write: int = 0,
    fusion_en: int = 0,
    nb_enable: int = 0,
) -> int:
    """Pack the current Shoot-and-Go rs1 layout.

    Bit allocation:
      [15:0]   constant_operand
      [31:16]  valid_row
      [32]     vector_compact_out
      [33]     vector_compact_in
      [35:34]  out_point
      [37:36]  input_point
      [38]     tile_strided_wr
      [40:39]  tile_strided_rd
      [41]     transpose_en_wr
      [43:42]  transpose_en_rd
      [44]     rope_en
      [46:45]  norm_mode
      [47]     act_en
      [49:48]  alu_mode
      [50]     mxu_en
      [55:51]  lut_write
      [56]     fusion_en
      [57]     nb_enable
      [63:58]  reserved = 0

    Note: the previous cache_enable field is not part of this ISA layout.
    """
    rs1 = 0
    rs1 |= (constant_operand & 0xFFFF) << 0
    rs1 |= (valid_row & 0xFFFF) << 16
    rs1 |= (vector_compact_out & 0x1) << 32
    rs1 |= (vector_compact_in & 0x1) << 33
    rs1 |= (out_point & 0x3) << 34
    rs1 |= (input_point & 0x3) << 36
    rs1 |= (tile_strided_wr & 0x1) << 38
    rs1 |= (tile_strided_rd & 0x3) << 39
    rs1 |= (transpose_en_wr & 0x1) << 41
    rs1 |= (transpose_en_rd & 0x3) << 42

    # hw_enables [50:44]
    rs1 |= (rope_en & 0x1) << 44
    rs1 |= (norm_mode & 0x3) << 45
    rs1 |= (act_en & 0x1) << 47
    rs1 |= (alu_mode & 0x3) << 48
    rs1 |= (mxu_en & 0x1) << 50

    rs1 |= (lut_write & 0x1F) << 51
    rs1 |= (fusion_en & 0x1) << 56
    rs1 |= (nb_enable & 0x1) << 57

    # [63:58] remain zero.
    return rs1


# =============================================================================
# [2] rs2: DRAM-resident npu_ctrl task descriptor
# =============================================================================

def build_rs2_struct(
    # 1. Data pointers
    input_addr: int = 0,
    weight1_addr: int = 0,
    weight2_addr: int = 0,
    # 2. Parameter pointers
    quant_param_addr: int = 0,
    angle_param_addr: int = 0,
    # 3. LUT pointers
    act_lut_addr: int = 0,
    exp_lut_addr: int = 0,
    scale_lut_addr: int = 0,
    rope_sin_addr: int = 0,
    rope_cos_addr: int = 0,
    # 4. Output pointers
    output_addr: int = 0,
    norm_buff_addr: int = 0,
    # 5. Dimensions: all in 1/16 scale
    out_rowNum: int = 0,
    out_intermNum: int = 0,
    out_colNum: int = 0,
    # Explicit tile totals
    input_total_tiles: int = 0,
    weight_total_tiles: int = 0,
    out_total_tiles: int = 0,
    # 6. Strided offsets: all in 1/256 scale
    input_offset: int = 0,
    weight_offset: int = 0,
    output_offset: int = 0,
) -> bytes:
    """Serialize the current npu_ctrl field list as packed little-endian bytes.

    Layout = 12 pointers + 9 uint32 fields = 132 bytes exactly.

    Host-side C must use the same packed/manual serialization contract if
    sizeof(npu_ctrl) is expected to be exactly 132 bytes; a normal LP64 C
    compiler may add tail padding otherwise.
    """
    qwords = [
        input_addr, weight1_addr, weight2_addr,
        quant_param_addr, angle_param_addr,
        act_lut_addr, exp_lut_addr, scale_lut_addr,
        rope_sin_addr, rope_cos_addr,
        output_addr, norm_buff_addr,
    ]
    dwords = [
        out_rowNum, out_intermNum, out_colNum,
        input_total_tiles, weight_total_tiles, out_total_tiles,
        input_offset, weight_offset, output_offset,
    ]

    qwords = [_u64(v, f"ptr[{i}]") for i, v in enumerate(qwords)]
    dwords = [_u32(v, f"u32[{i}]") for i, v in enumerate(dwords)]
    return struct.pack(RS2_FMT, *qwords, *dwords)


def gemm_tile_totals(row_tiles: int, k_tiles: int, col_tiles: int) -> Tuple[int, int, int]:
    """ISA-defined GEMM tile totals."""
    return (
        row_tiles * k_tiles,
        k_tiles * col_tiles,
        row_tiles * col_tiles,
    )


def vector_tile_totals(row_tiles: int, col_tiles: int, *, has_weight_vector: bool = False) -> Tuple[int, int, int]:
    """Convenience convention for the examples below.

    The ISA page gives explicit total-tile formulae for GEMM. For standalone
    vector/VPU operations, this compiler emits the physically consumed input
    and output tile counts directly. A norm weight vector, when used, is counted
    as one 16-element line per col tile.
    """
    vector_tiles = row_tiles * col_tiles
    weight_tiles = col_tiles if has_weight_vector else 0
    return vector_tiles, weight_tiles, vector_tiles


@dataclass(frozen=True)
class EncodedInstruction:
    name: str
    rs1: int
    rs2: bytes

    def __post_init__(self):
        if len(self.rs2) != RS2_SIZE:
            raise ValueError(f"{self.name}: rs2 descriptor must be {RS2_SIZE} bytes")


# =============================================================================
# [3] Example memory map
# =============================================================================

# Activations / intermediate data
MEM_IN_H_L        = 0x0000
MEM_NORM_OUT      = 0x1000
MEM_Q_OUT         = 0x2000
MEM_K_OUT         = 0x3000
MEM_V_OUT         = 0x4000
MEM_ATTN_OUT      = 0x5000
MEM_H_MID         = 0x6000
MEM_MLP_NORM_OUT  = 0x7000
MEM_GEGLU_OUT     = 0xA000
MEM_H_OUT         = 0xB000
MEM_NORM_BUF      = 0xC000
MEM_POST_NORM_BUF = 0xD000

# Weights
WT_NORM_IN        = 0x100000
WT_Q              = 0x110000
WT_K              = 0x120000
WT_V              = 0x130000
WT_O              = 0x140000
WT_NORM_POST      = 0x150000
WT_GATE           = 0x160000
WT_UP             = 0x170000
WT_DOWN           = 0x180000

# Parameter/LUT addresses are intentionally left as zero placeholders here.
# The real compiler should fill these from calibrated artifacts / GGUF layout.
QP_Q = QP_K = QP_V = QP_O = QP_GATE = QP_UP = QP_DOWN = 0
ACT_LUT_GELU = 0
ANGLE_PARAM = ROPE_SIN_LUT = ROPE_COS_LUT = 0


# =============================================================================
# [4] Single-layer compiler example
# =============================================================================

def compile_single_layer(
    *,
    valid_row: int = 0x0001,        # GEMV M=1 example: only logical row 0 is valid
    geglu_scale_operand: int = 0,   # GPALU scalar scaling encoding; 0 = disabled/not supplied
) -> List[EncodedInstruction]:
    instructions: List[EncodedInstruction] = []

    # Gemma-2B example dimensions in ISA 1/16 units.
    row_tiles = 1
    dim_d = 2048 // 16       # 128
    dim_mid = 16384 // 16    # 1024

    # -------------------------------------------------------------------------
    # Step 1. Input RMSNorm (one Shoot-and-Go task; internal phase sequencing)
    # -------------------------------------------------------------------------
    in_t, wt_t, out_t = vector_tile_totals(row_tiles, dim_d, has_weight_vector=True)
    rs1 = build_rs1(
        norm_mode=NORM_RMS,
        input_point=INPUT_VPU2,
        out_point=OUT_VPU2,
        nb_enable=1,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    rs2 = build_rs2_struct(
        input_addr=MEM_IN_H_L,
        weight1_addr=WT_NORM_IN,
        output_addr=MEM_NORM_OUT,
        norm_buff_addr=MEM_NORM_BUF,
        out_rowNum=row_tiles,
        out_colNum=dim_d,
        input_total_tiles=in_t,
        weight_total_tiles=wt_t,
        out_total_tiles=out_t,
    )
    instructions.append(EncodedInstruction("input_rmsnorm", rs1, rs2))

    # -------------------------------------------------------------------------
    # Step 2. Q / K / V projections
    # -------------------------------------------------------------------------
    gemm_in, gemm_w, gemm_out = gemm_tile_totals(row_tiles, dim_d, dim_d)
    for name, weight, output, qparam in (
        ("q_proj", WT_Q, MEM_Q_OUT, QP_Q),
        ("k_proj", WT_K, MEM_K_OUT, QP_K),
        ("v_proj", WT_V, MEM_V_OUT, QP_V),
    ):
        rs1 = build_rs1(
            mxu_en=1,
            input_point=INPUT_MXU,
            out_point=OUT_VPU1,
            valid_row=valid_row,
            vector_compact_in=1,
            vector_compact_out=1,
        )
        rs2 = build_rs2_struct(
            input_addr=MEM_NORM_OUT,
            weight1_addr=weight,
            quant_param_addr=qparam,
            output_addr=output,
            out_rowNum=row_tiles,
            out_intermNum=dim_d,
            out_colNum=dim_d,
            input_total_tiles=gemm_in,
            weight_total_tiles=gemm_w,
            out_total_tiles=gemm_out,
        )
        instructions.append(EncodedInstruction(name, rs1, rs2))

    # -------------------------------------------------------------------------
    # Step 3. RoPE for Q and K (kept as standalone tasks in this example)
    # -------------------------------------------------------------------------
    rope_in, _, rope_out = vector_tile_totals(row_tiles, dim_d)
    rope_rs1 = build_rs1(
        rope_en=1,
        input_point=INPUT_VPU2,
        out_point=OUT_VPU2,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    for name, address in (("q_rope", MEM_Q_OUT), ("k_rope", MEM_K_OUT)):
        rs2 = build_rs2_struct(
            input_addr=address,
            angle_param_addr=ANGLE_PARAM,
            rope_sin_addr=ROPE_SIN_LUT,
            rope_cos_addr=ROPE_COS_LUT,
            output_addr=address,
            out_rowNum=row_tiles,
            out_colNum=dim_d,
            input_total_tiles=rope_in,
            weight_total_tiles=0,
            out_total_tiles=rope_out,
        )
        instructions.append(EncodedInstruction(name, rope_rs1, rs2))

    # -------------------------------------------------------------------------
    # Attention score / softmax / V aggregation omitted here, same as old sample.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Step 4. O projection + residual add
    # weight2_addr carries the second VPU1 operand source descriptor.
    # -------------------------------------------------------------------------
    rs1 = build_rs1(
        mxu_en=1,
        alu_mode=ALU_ADD,
        input_point=INPUT_MXU,
        out_point=OUT_VPU1,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    rs2 = build_rs2_struct(
        input_addr=MEM_ATTN_OUT,
        weight1_addr=WT_O,
        weight2_addr=MEM_IN_H_L,
        quant_param_addr=QP_O,
        output_addr=MEM_H_MID,
        out_rowNum=row_tiles,
        out_intermNum=dim_d,
        out_colNum=dim_d,
        input_total_tiles=gemm_in,
        weight_total_tiles=gemm_w,
        out_total_tiles=gemm_out,
    )
    instructions.append(EncodedInstruction("o_proj_residual", rs1, rs2))

    # -------------------------------------------------------------------------
    # Step 5. Post-attention RMSNorm
    # -------------------------------------------------------------------------
    in_t, wt_t, out_t = vector_tile_totals(row_tiles, dim_d, has_weight_vector=True)
    rs1 = build_rs1(
        norm_mode=NORM_RMS,
        input_point=INPUT_VPU2,
        out_point=OUT_VPU2,
        nb_enable=1,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    rs2 = build_rs2_struct(
        input_addr=MEM_H_MID,
        weight1_addr=WT_NORM_POST,
        output_addr=MEM_MLP_NORM_OUT,
        norm_buff_addr=MEM_POST_NORM_BUF,
        out_rowNum=row_tiles,
        out_colNum=dim_d,
        input_total_tiles=in_t,
        weight_total_tiles=wt_t,
        out_total_tiles=out_t,
    )
    instructions.append(EncodedInstruction("post_rmsnorm", rs1, rs2))

    # -------------------------------------------------------------------------
    # Step 6. Fused GeGLU Shoot-and-Go task
    #   GeLU(X * W_gate^T) ** (X * W_up^T)
    #
    # The Sequencer uses weight1/weight2 as the two GEMM branches.  act_en marks
    # the activation-bearing branch, alu_mode=MUL requests the elementwise merge,
    # and fusion_en turns the multi-branch task on.
    #
    # constant_operand is reserved here for the assumed GPALU scalar-scale path.
    # Its exact fixed-point encoding should follow the final GPALU implementation.
    # -------------------------------------------------------------------------
    gate_in, gate_w, gate_out = gemm_tile_totals(row_tiles, dim_d, dim_mid)
    rs1 = build_rs1(
        constant_operand=geglu_scale_operand,
        mxu_en=1,
        act_en=1,
        alu_mode=ALU_MUL,
        fusion_en=1,
        input_point=INPUT_MXU,
        out_point=OUT_VPU1,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    rs2 = build_rs2_struct(
        input_addr=MEM_MLP_NORM_OUT,
        weight1_addr=WT_GATE,
        weight2_addr=WT_UP,
        quant_param_addr=QP_GATE,  # real compiler may pack/point to both branch params
        act_lut_addr=ACT_LUT_GELU,
        output_addr=MEM_GEGLU_OUT,
        out_rowNum=row_tiles,
        out_intermNum=dim_d,
        out_colNum=dim_mid,
        input_total_tiles=gate_in,
        weight_total_tiles=gate_w,
        out_total_tiles=gate_out,
    )
    instructions.append(EncodedInstruction("fused_geglu", rs1, rs2))

    # -------------------------------------------------------------------------
    # Step 7. Down projection + second residual
    # -------------------------------------------------------------------------
    down_in, down_w, down_out = gemm_tile_totals(row_tiles, dim_mid, dim_d)
    rs1 = build_rs1(
        mxu_en=1,
        alu_mode=ALU_ADD,
        input_point=INPUT_MXU,
        out_point=OUT_VPU1,
        valid_row=valid_row,
        vector_compact_in=1,
        vector_compact_out=1,
    )
    rs2 = build_rs2_struct(
        input_addr=MEM_GEGLU_OUT,
        weight1_addr=WT_DOWN,
        weight2_addr=MEM_H_MID,
        quant_param_addr=QP_DOWN,
        output_addr=MEM_H_OUT,
        out_rowNum=row_tiles,
        out_intermNum=dim_mid,
        out_colNum=dim_d,
        input_total_tiles=down_in,
        weight_total_tiles=down_w,
        out_total_tiles=down_out,
    )
    instructions.append(EncodedInstruction("down_proj_residual", rs1, rs2))

    return instructions


# =============================================================================
# [5] Program serialization / smoke test
# =============================================================================

def write_program(path: str, instructions: List[EncodedInstruction]) -> None:
    """Write an offline compiler image: [rs1:uint64][npu_ctrl:132B] per task.

    At runtime, the host places each npu_ctrl in DRAM and passes its address as
    RoCC rs2.  This flat file is an offline packaging format, not the literal
    electrical value carried by the RoCC rs2 register.
    """
    with open(path, "wb") as f:
        for inst in instructions:
            f.write(struct.pack("<Q", inst.rs1))
            f.write(inst.rs2)


if __name__ == "__main__":
    layer_instrs = compile_single_layer()

    # ISA sanity checks
    assert all((inst.rs1 >> 58) == 0 for inst in layer_instrs), "rs1 reserved bits must be zero"
    assert all(len(inst.rs2) == 132 for inst in layer_instrs)

    output_path = "gemma_layer0.bin"
    write_program(output_path, layer_instrs)

    print(f"npu_ctrl packed size: {RS2_SIZE} bytes")
    print(f"instructions: {len(layer_instrs)}")
    for i, inst in enumerate(layer_instrs):
        print(f"  {i:02d} {inst.name:20s} rs1=0x{inst.rs1:016X} rs2={len(inst.rs2)}B")
    print(f"wrote: {output_path}")
