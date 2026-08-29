#!/usr/bin/env ruby
# Regenerates the Metal kernels: the portable operators translated from the
# OpenCL implementation, followed by src/matrix.metal verbatim. Keep
# operator bodies in one place; this script only translates address spaces,
# launch built-ins, vector loads/stores, and Metal's explicit buffer bindings.

input, output = ARGV
abort "usage: port-metal.rb INPUT.cl OUTPUT.metal" unless input && output
source = File.read(input)
source = source.split("// ---------------------------------------------------------------------------\n// Matrix multiplication on the Xe matrix engines", 2).first
body = source[source.index("enum Binary")..]

prologue = <<~'METAL'
  //! Portable SAM 3 operators for Apple GPUs. Generated from kernels.cl by
  //! tools/port-metal.rb; edit the OpenCL operator bodies, then regenerate.
  #include <metal_stdlib>
  using namespace metal;

  #if SAM3_HALF
  typedef half real;
  typedef half4 realv;
  #define TO_REALV(v) half4(v)
  #else
  typedef float real;
  typedef float4 realv;
  #define TO_REALV(v) (v)
  #endif
  typedef float4 floatv;
  #define LANE_STEP 4
  #define TO_FLOATV(v) float4(v)
  #define VLOADV(i, p) (((device const realv*)(p))[i])
  #define VSTOREV(v, i, p) (((device realv*)(p))[i] = (v))
  #define INFINITY __builtin_inff()
  #define CLK_LOCAL_MEM_FENCE 0
  #define barrier(ignore) threadgroup_barrier(mem_flags::mem_threadgroup)
  #define get_global_id(axis) gid[axis]
  #define get_local_id(axis) lid[axis]
  #define get_group_id(axis) group_id[axis]
  #define get_local_size(axis) threads_per_group[axis]

METAL

body.gsub!(/#if defined\(cl_khr_subgroups\).*?#endif\n/m, "")
# The two row reductions, which OpenCL spells against sub-groups where the
# driver has them and a local memory tree where it does not. Metal always has
# the SIMD group, so neither branch is what it wants; both go, and the
# prologue's own pair takes their place.
metal_reductions = <<~'METAL'
  /// A whole SIMD group folds in one instruction and needs no barrier, which
  /// leaves only the fold across the groups of a work group -- two barriers for
  /// the entire reduction rather than one per halving. The leading barrier is
  /// what makes `scratch` reusable between two reductions in a row: it retires
  /// the reads the previous one ended with.
  #define REDUCE_ACROSS_SIMD_GROUPS(combine, value, scratch, identity)     \
      float within = value;                                               \
      uint groups = block_size / 32u;                                      \
      if (groups <= 1u) return within;                                     \
      barrier(CLK_LOCAL_MEM_FENCE);                                        \
      if ((lane % 32u) == 0u) scratch[lane / 32u] = within;                \
      barrier(CLK_LOCAL_MEM_FENCE);                                        \
      float folded = identity;                                             \
      for (uint g = 0; g < groups; g++) folded = combine(folded, scratch[g]); \
      return folded;

  inline float addf(float a, float b) { return a + b; }

  inline float reduceSumLocal(float val, threadgroup float* scratch, uint lane, uint block_size) {
      REDUCE_ACROSS_SIMD_GROUPS(addf, simd_sum(val), scratch, 0.0f)
  }

  inline float reduceMaxLocal(float val, threadgroup float* scratch, uint lane, uint block_size) {
      REDUCE_ACROSS_SIMD_GROUPS(fmax, simd_max(val), scratch, -INFINITY)
  }
METAL

body.sub!(/#if REDUCE_BY_SUB_GROUP.*?\n#endif\n/m, metal_reductions)
body.gsub!(/^#endif\n/, "")
body.gsub!("__global ", "device ")
body.gsub!("__local ", "threadgroup ")
body.gsub!("__private ", "thread ")
body.gsub!("__kernel ", "kernel ")
body.gsub!(/\bthread\b/, "thread_index")
body.gsub!(/\(\(float\*\)&([A-Za-z_][A-Za-z0-9_]*)\)\[([^\]]+)\]/, '\\1[\\2]')
# OpenCL numbers a vector's components; Metal names them.
body.gsub!(/\.s([0-3])\b/) { ".#{%w[x y z w][Regexp.last_match(1).to_i]}" }
body.gsub!(/\(floatv\)\(([^\n;]*)\)/, 'floatv(\\1)')
body.gsub!(/\(realv\)\(([^\n;]*)\)/, 'realv(\\1)')
body.gsub!(/#define\s+fastExp\(x\)\s+native_exp\(x\)\n/, "")
body.gsub!("fastExp", "exp")

body.gsub!(/kernel void (\w+)\s*\((.*?)\n\)\s*\{/m) do
  name = Regexp.last_match(1)
  raw = Regexp.last_match(2)
  params = raw.split(",").map(&:strip)
  bound = params.each_with_index.map do |param, index|
    if param.include?("device ")
      "    #{param} [[buffer(#{index})]]"
    else
      type, scalar_name = param.sub(/^(const )?/, '').split(/\s+(?=[A-Za-z_][A-Za-z0-9_]*$)/, 2)
      "    constant #{type}& #{scalar_name} [[buffer(#{index})]]"
    end
  end
  builtins = [
    "    uint3 gid [[thread_position_in_grid]]",
    "    uint3 lid [[thread_position_in_threadgroup]]",
    "    uint3 group_id [[threadgroup_position_in_grid]]",
    "    uint3 threads_per_group [[threads_per_threadgroup]]",
  ]
  "kernel void #{name}(\n#{(bound + builtins).join(",\n")}\n) {"
end

# The matrix-unit kernels are Metal's own, the way the Xe ones are OpenCL's.
epilogue = File.read(File.join(File.dirname(input), "matrix.metal"))

File.write(output, prologue + body + epilogue)
