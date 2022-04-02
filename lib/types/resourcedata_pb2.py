# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lib/types/resourcedata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.common import base_pb2 as lib_dot_common_dot_base__pb2
from lib.common import data_pb2 as lib_dot_common_dot_data__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lib/types/resourcedata.proto',
  package='types',
  syntax='proto3',
  serialized_options=b'Z+github.com/RosettaFlow/Carrier-Go/lib/types',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x1clib/types/resourcedata.proto\x12\x05types\x1a\x15lib/common/base.proto\x1a\x15lib/common/data.proto\"\x83\x03\n\nResourcePB\x12\x13\n\x0bidentity_id\x18\x01 \x01(\t\x12\x0f\n\x07node_id\x18\x02 \x01(\t\x12\x11\n\tnode_name\x18\x03 \x01(\t\x12\x0f\n\x07\x64\x61ta_id\x18\x04 \x01(\t\x12-\n\x0b\x64\x61ta_status\x18\x05 \x01(\x0e\x32\x18.api.protobuf.DataStatus\x12\'\n\x05state\x18\x06 \x01(\x0e\x32\x18.api.protobuf.PowerState\x12\x11\n\ttotal_mem\x18\x07 \x01(\x04\x12\x10\n\x08used_mem\x18\x08 \x01(\x04\x12\x17\n\x0ftotal_processor\x18\t \x01(\r\x12\x16\n\x0eused_processor\x18\n \x01(\r\x12\x17\n\x0ftotal_bandwidth\x18\x0b \x01(\x04\x12\x16\n\x0eused_bandwidth\x18\x0c \x01(\x04\x12\x12\n\ntotal_disk\x18\r \x01(\x04\x12\x11\n\tused_disk\x18\x0e \x01(\x04\x12\x12\n\npublish_at\x18\x10 \x01(\x04\x12\x11\n\tupdate_at\x18\x11 \x01(\x04\"\xf6\x02\n\x0fLocalResourcePB\x12\x13\n\x0bidentity_id\x18\x01 \x01(\t\x12\x0f\n\x07node_id\x18\x02 \x01(\t\x12\x11\n\tnode_name\x18\x03 \x01(\t\x12\x13\n\x0bjob_node_id\x18\x04 \x01(\t\x12\x0f\n\x07\x64\x61ta_id\x18\x05 \x01(\t\x12-\n\x0b\x64\x61ta_status\x18\x06 \x01(\x0e\x32\x18.api.protobuf.DataStatus\x12\'\n\x05state\x18\x07 \x01(\x0e\x32\x18.api.protobuf.PowerState\x12\x11\n\ttotal_mem\x18\x08 \x01(\x04\x12\x10\n\x08used_mem\x18\t \x01(\x04\x12\x17\n\x0ftotal_processor\x18\n \x01(\r\x12\x16\n\x0eused_processor\x18\x0b \x01(\r\x12\x17\n\x0ftotal_bandwidth\x18\x0c \x01(\x04\x12\x16\n\x0eused_bandwidth\x18\r \x01(\x04\x12\x12\n\ntotal_disk\x18\x0e \x01(\x04\x12\x11\n\tused_disk\x18\x0f \x01(\x04\"\x8d\x01\n\x05Power\x12\x13\n\x0bjob_node_id\x18\x01 \x01(\t\x12\x10\n\x08power_id\x18\x02 \x01(\t\x12\x34\n\x0eusage_overview\x18\x03 \x01(\x0b\x32\x1c.types.ResourceUsageOverview\x12\'\n\x05state\x18\x04 \x01(\x0e\x32\x18.api.protobuf.PowerState\"\xec\x01\n\x10PowerUsageDetail\x12\x31\n\x0binformation\x18\x01 \x01(\x0b\x32\x1c.types.ResourceUsageOverview\x12\x18\n\x10total_task_count\x18\x02 \x01(\r\x12\x1a\n\x12\x63urrent_task_count\x18\x03 \x01(\r\x12\x1f\n\x05tasks\x18\x04 \x03(\x0b\x32\x10.types.PowerTask\x12\'\n\x05state\x18\x05 \x01(\x0e\x32\x18.api.protobuf.PowerState\x12\x12\n\npublish_at\x18\x06 \x01(\x04\x12\x11\n\tupdate_at\x18\x07 \x01(\x04\"\xc8\x02\n\tPowerTask\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x11\n\ttask_name\x18\x02 \x01(\t\x12)\n\x05owner\x18\x03 \x01(\x0b\x32\x1a.api.protobuf.Organization\x12,\n\x08partners\x18\x04 \x03(\x0b\x32\x1a.api.protobuf.Organization\x12-\n\treceivers\x18\x05 \x03(\x0b\x32\x1a.api.protobuf.Organization\x12=\n\x0eoperation_cost\x18\x06 \x01(\x0b\x32%.api.protobuf.TaskResourceCostDeclare\x12>\n\x0foperation_spend\x18\x07 \x01(\x0b\x32%.api.protobuf.TaskResourceCostDeclare\x12\x10\n\x08\x63reateAt\x18\x08 \x01(\x04\"\xc5\x01\n\x15ResourceUsageOverview\x12\x11\n\ttotal_mem\x18\x01 \x01(\x04\x12\x10\n\x08used_mem\x18\x02 \x01(\x04\x12\x17\n\x0ftotal_processor\x18\x03 \x01(\r\x12\x16\n\x0eused_processor\x18\x04 \x01(\r\x12\x17\n\x0ftotal_bandwidth\x18\x05 \x01(\x04\x12\x16\n\x0eused_bandwidth\x18\x06 \x01(\x04\x12\x12\n\ntotal_disk\x18\x07 \x01(\x04\x12\x11\n\tused_disk\x18\x08 \x01(\x04\x42-Z+github.com/RosettaFlow/Carrier-Go/lib/typesb\x06proto3'
  ,
  dependencies=[lib_dot_common_dot_base__pb2.DESCRIPTOR,lib_dot_common_dot_data__pb2.DESCRIPTOR,])




_RESOURCEPB = _descriptor.Descriptor(
  name='ResourcePB',
  full_name='types.ResourcePB',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity_id', full_name='types.ResourcePB.identity_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_id', full_name='types.ResourcePB.node_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_name', full_name='types.ResourcePB.node_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_id', full_name='types.ResourcePB.data_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_status', full_name='types.ResourcePB.data_status', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='types.ResourcePB.state', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_mem', full_name='types.ResourcePB.total_mem', index=6,
      number=7, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_mem', full_name='types.ResourcePB.used_mem', index=7,
      number=8, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_processor', full_name='types.ResourcePB.total_processor', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_processor', full_name='types.ResourcePB.used_processor', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_bandwidth', full_name='types.ResourcePB.total_bandwidth', index=10,
      number=11, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_bandwidth', full_name='types.ResourcePB.used_bandwidth', index=11,
      number=12, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_disk', full_name='types.ResourcePB.total_disk', index=12,
      number=13, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_disk', full_name='types.ResourcePB.used_disk', index=13,
      number=14, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='publish_at', full_name='types.ResourcePB.publish_at', index=14,
      number=16, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='update_at', full_name='types.ResourcePB.update_at', index=15,
      number=17, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=86,
  serialized_end=473,
)


_LOCALRESOURCEPB = _descriptor.Descriptor(
  name='LocalResourcePB',
  full_name='types.LocalResourcePB',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity_id', full_name='types.LocalResourcePB.identity_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_id', full_name='types.LocalResourcePB.node_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='node_name', full_name='types.LocalResourcePB.node_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='job_node_id', full_name='types.LocalResourcePB.job_node_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_id', full_name='types.LocalResourcePB.data_id', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_status', full_name='types.LocalResourcePB.data_status', index=5,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='types.LocalResourcePB.state', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_mem', full_name='types.LocalResourcePB.total_mem', index=7,
      number=8, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_mem', full_name='types.LocalResourcePB.used_mem', index=8,
      number=9, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_processor', full_name='types.LocalResourcePB.total_processor', index=9,
      number=10, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_processor', full_name='types.LocalResourcePB.used_processor', index=10,
      number=11, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_bandwidth', full_name='types.LocalResourcePB.total_bandwidth', index=11,
      number=12, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_bandwidth', full_name='types.LocalResourcePB.used_bandwidth', index=12,
      number=13, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_disk', full_name='types.LocalResourcePB.total_disk', index=13,
      number=14, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_disk', full_name='types.LocalResourcePB.used_disk', index=14,
      number=15, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=476,
  serialized_end=850,
)


_POWER = _descriptor.Descriptor(
  name='Power',
  full_name='types.Power',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_node_id', full_name='types.Power.job_node_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='power_id', full_name='types.Power.power_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='usage_overview', full_name='types.Power.usage_overview', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='types.Power.state', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=853,
  serialized_end=994,
)


_POWERUSAGEDETAIL = _descriptor.Descriptor(
  name='PowerUsageDetail',
  full_name='types.PowerUsageDetail',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='information', full_name='types.PowerUsageDetail.information', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_task_count', full_name='types.PowerUsageDetail.total_task_count', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='current_task_count', full_name='types.PowerUsageDetail.current_task_count', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tasks', full_name='types.PowerUsageDetail.tasks', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='types.PowerUsageDetail.state', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='publish_at', full_name='types.PowerUsageDetail.publish_at', index=5,
      number=6, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='update_at', full_name='types.PowerUsageDetail.update_at', index=6,
      number=7, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=997,
  serialized_end=1233,
)


_POWERTASK = _descriptor.Descriptor(
  name='PowerTask',
  full_name='types.PowerTask',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='types.PowerTask.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_name', full_name='types.PowerTask.task_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='owner', full_name='types.PowerTask.owner', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='partners', full_name='types.PowerTask.partners', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='receivers', full_name='types.PowerTask.receivers', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='operation_cost', full_name='types.PowerTask.operation_cost', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='operation_spend', full_name='types.PowerTask.operation_spend', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='createAt', full_name='types.PowerTask.createAt', index=7,
      number=8, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1236,
  serialized_end=1564,
)


_RESOURCEUSAGEOVERVIEW = _descriptor.Descriptor(
  name='ResourceUsageOverview',
  full_name='types.ResourceUsageOverview',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='total_mem', full_name='types.ResourceUsageOverview.total_mem', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_mem', full_name='types.ResourceUsageOverview.used_mem', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_processor', full_name='types.ResourceUsageOverview.total_processor', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_processor', full_name='types.ResourceUsageOverview.used_processor', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_bandwidth', full_name='types.ResourceUsageOverview.total_bandwidth', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_bandwidth', full_name='types.ResourceUsageOverview.used_bandwidth', index=5,
      number=6, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='total_disk', full_name='types.ResourceUsageOverview.total_disk', index=6,
      number=7, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='used_disk', full_name='types.ResourceUsageOverview.used_disk', index=7,
      number=8, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1567,
  serialized_end=1764,
)

_RESOURCEPB.fields_by_name['data_status'].enum_type = lib_dot_common_dot_base__pb2._DATASTATUS
_RESOURCEPB.fields_by_name['state'].enum_type = lib_dot_common_dot_base__pb2._POWERSTATE
_LOCALRESOURCEPB.fields_by_name['data_status'].enum_type = lib_dot_common_dot_base__pb2._DATASTATUS
_LOCALRESOURCEPB.fields_by_name['state'].enum_type = lib_dot_common_dot_base__pb2._POWERSTATE
_POWER.fields_by_name['usage_overview'].message_type = _RESOURCEUSAGEOVERVIEW
_POWER.fields_by_name['state'].enum_type = lib_dot_common_dot_base__pb2._POWERSTATE
_POWERUSAGEDETAIL.fields_by_name['information'].message_type = _RESOURCEUSAGEOVERVIEW
_POWERUSAGEDETAIL.fields_by_name['tasks'].message_type = _POWERTASK
_POWERUSAGEDETAIL.fields_by_name['state'].enum_type = lib_dot_common_dot_base__pb2._POWERSTATE
_POWERTASK.fields_by_name['owner'].message_type = lib_dot_common_dot_base__pb2._ORGANIZATION
_POWERTASK.fields_by_name['partners'].message_type = lib_dot_common_dot_base__pb2._ORGANIZATION
_POWERTASK.fields_by_name['receivers'].message_type = lib_dot_common_dot_base__pb2._ORGANIZATION
_POWERTASK.fields_by_name['operation_cost'].message_type = lib_dot_common_dot_data__pb2._TASKRESOURCECOSTDECLARE
_POWERTASK.fields_by_name['operation_spend'].message_type = lib_dot_common_dot_data__pb2._TASKRESOURCECOSTDECLARE
DESCRIPTOR.message_types_by_name['ResourcePB'] = _RESOURCEPB
DESCRIPTOR.message_types_by_name['LocalResourcePB'] = _LOCALRESOURCEPB
DESCRIPTOR.message_types_by_name['Power'] = _POWER
DESCRIPTOR.message_types_by_name['PowerUsageDetail'] = _POWERUSAGEDETAIL
DESCRIPTOR.message_types_by_name['PowerTask'] = _POWERTASK
DESCRIPTOR.message_types_by_name['ResourceUsageOverview'] = _RESOURCEUSAGEOVERVIEW
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ResourcePB = _reflection.GeneratedProtocolMessageType('ResourcePB', (_message.Message,), {
  'DESCRIPTOR' : _RESOURCEPB,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.ResourcePB)
  })
_sym_db.RegisterMessage(ResourcePB)

LocalResourcePB = _reflection.GeneratedProtocolMessageType('LocalResourcePB', (_message.Message,), {
  'DESCRIPTOR' : _LOCALRESOURCEPB,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.LocalResourcePB)
  })
_sym_db.RegisterMessage(LocalResourcePB)

Power = _reflection.GeneratedProtocolMessageType('Power', (_message.Message,), {
  'DESCRIPTOR' : _POWER,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.Power)
  })
_sym_db.RegisterMessage(Power)

PowerUsageDetail = _reflection.GeneratedProtocolMessageType('PowerUsageDetail', (_message.Message,), {
  'DESCRIPTOR' : _POWERUSAGEDETAIL,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.PowerUsageDetail)
  })
_sym_db.RegisterMessage(PowerUsageDetail)

PowerTask = _reflection.GeneratedProtocolMessageType('PowerTask', (_message.Message,), {
  'DESCRIPTOR' : _POWERTASK,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.PowerTask)
  })
_sym_db.RegisterMessage(PowerTask)

ResourceUsageOverview = _reflection.GeneratedProtocolMessageType('ResourceUsageOverview', (_message.Message,), {
  'DESCRIPTOR' : _RESOURCEUSAGEOVERVIEW,
  '__module__' : 'lib.types.resourcedata_pb2'
  # @@protoc_insertion_point(class_scope:types.ResourceUsageOverview)
  })
_sym_db.RegisterMessage(ResourceUsageOverview)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)