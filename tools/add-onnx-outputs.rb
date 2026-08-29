# Diagnostic helper: append inferred float graph outputs without rewriting the
# rest of the protobuf. ONNX Runtime can then expose otherwise-internal values.
def varint(value)
  bytes = +"".b
  loop do
    byte = value & 0x7f
    value >>= 7
    bytes << (value.zero? ? byte : byte | 0x80)
    return bytes if value.zero?
  end
end

def read_varint(data, offset)
  value = 0
  shift = 0
  loop do
    byte = data.getbyte(offset)
    offset += 1
    value |= (byte & 0x7f) << shift
    return [value, offset] if (byte & 0x80).zero?
    shift += 7
  end
end

input, output, *names = ARGV
abort "usage: add-onnx-outputs.rb INPUT OUTPUT NAME..." unless input && output && !names.empty?
model = File.binread(input)
offset = 0
rebuilt = +"".b
found = false
while offset < model.bytesize
  field_start = offset
  key, offset = read_varint(model, offset)
  wire = key & 7
  if wire == 2
    length, payload_at = read_varint(model, offset)
    payload_end = payload_at + length
    if (key >> 3) == 7
      graph = model.byteslice(payload_at, length).dup
      names.each do |name|
        # 3D dynamic shape
        dim = "\x0a\x02\x12\x00".b
        shape = dim + dim + dim
        tensor_type = "\x08\x01\x12".b + varint(shape.bytesize) + shape
        type = "\x0a".b + varint(tensor_type.bytesize) + tensor_type
        info = "\x0a".b + varint(name.bytesize) + name.b + "\x12".b + varint(type.bytesize) + type
        graph << varint((12 << 3) | 2) << varint(info.bytesize) << info
      end
      rebuilt << model.byteslice(field_start, offset - field_start)
      rebuilt << varint(graph.bytesize) << graph
      found = true
    else
      rebuilt << model.byteslice(field_start, payload_end - field_start)
    end
    offset = payload_end
  else
    abort "unsupported top-level protobuf wire type #{wire}" unless [0, 1, 5].include?(wire)
    if wire == 0
      _, offset = read_varint(model, offset)
    else
      offset += wire == 1 ? 8 : 4
    end
    rebuilt << model.byteslice(field_start, offset - field_start)
  end
end
abort "model graph not found" unless found
File.binwrite(output, rebuilt)
