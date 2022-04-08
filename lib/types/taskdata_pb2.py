# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lib/types/taskdata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.types import base_pb2 as lib_dot_types_dot_base__pb2
from lib.types import resourcedata_pb2 as lib_dot_types_dot_resourcedata__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lib/types/taskdata.proto',
  package='types',
  syntax='proto3',
  serialized_options=b'Z+github.com/RosettaFlow/Carrier-Go/lib/types',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x18lib/types/taskdata.proto\x12\x05types\x1a\x14lib/types/base.proto\x1a\x1clib/types/resourcedata.proto\"\xb5\x07\n\x06TaskPB\x12\x0f\n\x07task_id\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x61ta_id\x18\x02 \x01(\t\x12&\n\x0b\x64\x61ta_status\x18\x03 \x01(\x0e\x32\x11.types.DataStatus\x12\x0c\n\x04user\x18\x04 \x01(\t\x12\"\n\tuser_type\x18\x05 \x01(\x0e\x32\x0f.types.UserType\x12\x11\n\ttask_name\x18\x06 \x01(\t\x12\'\n\x06sender\x18\x07 \x01(\x0b\x32\x17.types.TaskOrganization\x12.\n\ralgo_supplier\x18\x08 \x01(\x0b\x32\x17.types.TaskOrganization\x12/\n\x0e\x64\x61ta_suppliers\x18\t \x03(\x0b\x32\x17.types.TaskOrganization\x12\x30\n\x0fpower_suppliers\x18\n \x03(\x0b\x32\x17.types.TaskOrganization\x12*\n\treceivers\x18\x0b \x03(\x0b\x32\x17.types.TaskOrganization\x12\x18\n\x10\x64\x61ta_policy_type\x18\x0c \x01(\r\x12\x1a\n\x12\x64\x61ta_policy_option\x18\r \x01(\t\x12\x19\n\x11power_policy_type\x18\x0e \x01(\r\x12\x1b\n\x13power_policy_option\x18\x0f \x01(\t\x12\x1d\n\x15\x64\x61ta_flow_policy_type\x18\x10 \x01(\r\x12\x1f\n\x17\x64\x61ta_flow_policy_option\x18\x11 \x01(\t\x12\x36\n\x0eoperation_cost\x18\x12 \x01(\x0b\x32\x1e.types.TaskResourceCostDeclare\x12\x16\n\x0e\x61lgorithm_code\x18\x13 \x01(\t\x12\x19\n\x11meta_algorithm_id\x18\x14 \x01(\t\x12#\n\x1b\x61lgorithm_code_extra_params\x18\x15 \x01(\t\x12>\n\x16power_resource_options\x18\x16 \x03(\x0b\x32\x1e.types.TaskPowerResourceOption\x12\x1f\n\x05state\x18\x17 \x01(\x0e\x32\x10.types.TaskState\x12\x0e\n\x06reason\x18\x18 \x01(\t\x12\x0c\n\x04\x64\x65sc\x18\x19 \x01(\t\x12\x11\n\tcreate_at\x18\x1a \x01(\x04\x12\x0e\n\x06\x65nd_at\x18\x1b \x01(\x04\x12\x10\n\x08start_at\x18\x1c \x01(\x04\x12%\n\x0btask_events\x18\x1d \x03(\x0b\x32\x10.types.TaskEvent\x12\x0c\n\x04sign\x18\x1e \x01(\x0c\x12\r\n\x05nonce\x18\x1f \x01(\x04\"i\n\x17TaskPowerResourceOption\x12\x10\n\x08party_id\x18\x01 \x01(\t\x12<\n\x16resource_used_overview\x18\x02 \x01(\x0b\x32\x1c.types.ResourceUsageOverview\"u\n\tTaskEvent\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0f\n\x07task_id\x18\x02 \x01(\t\x12\x13\n\x0bidentity_id\x18\x03 \x01(\t\x12\x10\n\x08party_id\x18\x04 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x05 \x01(\t\x12\x11\n\tcreate_at\x18\x06 \x01(\x04\x42-Z+github.com/RosettaFlow/Carrier-Go/lib/typesb\x06proto3'
  ,
  dependencies=[lib_dot_types_dot_base__pb2.DESCRIPTOR,lib_dot_types_dot_resourcedata__pb2.DESCRIPTOR,])




_TASKPB = _descriptor.Descriptor(
  name='TaskPB',
  full_name='types.TaskPB',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_id', full_name='types.TaskPB.task_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_id', full_name='types.TaskPB.data_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_status', full_name='types.TaskPB.data_status', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user', full_name='types.TaskPB.user', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='user_type', full_name='types.TaskPB.user_type', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_name', full_name='types.TaskPB.task_name', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sender', full_name='types.TaskPB.sender', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algo_supplier', full_name='types.TaskPB.algo_supplier', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_suppliers', full_name='types.TaskPB.data_suppliers', index=8,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='power_suppliers', full_name='types.TaskPB.power_suppliers', index=9,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='receivers', full_name='types.TaskPB.receivers', index=10,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_policy_type', full_name='types.TaskPB.data_policy_type', index=11,
      number=12, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_policy_option', full_name='types.TaskPB.data_policy_option', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='power_policy_type', full_name='types.TaskPB.power_policy_type', index=13,
      number=14, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='power_policy_option', full_name='types.TaskPB.power_policy_option', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_flow_policy_type', full_name='types.TaskPB.data_flow_policy_type', index=15,
      number=16, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_flow_policy_option', full_name='types.TaskPB.data_flow_policy_option', index=16,
      number=17, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='operation_cost', full_name='types.TaskPB.operation_cost', index=17,
      number=18, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm_code', full_name='types.TaskPB.algorithm_code', index=18,
      number=19, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='meta_algorithm_id', full_name='types.TaskPB.meta_algorithm_id', index=19,
      number=20, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='algorithm_code_extra_params', full_name='types.TaskPB.algorithm_code_extra_params', index=20,
      number=21, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='power_resource_options', full_name='types.TaskPB.power_resource_options', index=21,
      number=22, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='types.TaskPB.state', index=22,
      number=23, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='reason', full_name='types.TaskPB.reason', index=23,
      number=24, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='desc', full_name='types.TaskPB.desc', index=24,
      number=25, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='create_at', full_name='types.TaskPB.create_at', index=25,
      number=26, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='end_at', full_name='types.TaskPB.end_at', index=26,
      number=27, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='start_at', full_name='types.TaskPB.start_at', index=27,
      number=28, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_events', full_name='types.TaskPB.task_events', index=28,
      number=29, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sign', full_name='types.TaskPB.sign', index=29,
      number=30, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='nonce', full_name='types.TaskPB.nonce', index=30,
      number=31, type=4, cpp_type=4, label=1,
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
  serialized_start=88,
  serialized_end=1037,
)


_TASKPOWERRESOURCEOPTION = _descriptor.Descriptor(
  name='TaskPowerResourceOption',
  full_name='types.TaskPowerResourceOption',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='party_id', full_name='types.TaskPowerResourceOption.party_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='resource_used_overview', full_name='types.TaskPowerResourceOption.resource_used_overview', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1039,
  serialized_end=1144,
)


_TASKEVENT = _descriptor.Descriptor(
  name='TaskEvent',
  full_name='types.TaskEvent',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='types.TaskEvent.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_id', full_name='types.TaskEvent.task_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='identity_id', full_name='types.TaskEvent.identity_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='party_id', full_name='types.TaskEvent.party_id', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='content', full_name='types.TaskEvent.content', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='create_at', full_name='types.TaskEvent.create_at', index=5,
      number=6, type=4, cpp_type=4, label=1,
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
  serialized_start=1146,
  serialized_end=1263,
)

_TASKPB.fields_by_name['data_status'].enum_type = lib_dot_types_dot_base__pb2._DATASTATUS
_TASKPB.fields_by_name['user_type'].enum_type = lib_dot_types_dot_base__pb2._USERTYPE
_TASKPB.fields_by_name['sender'].message_type = lib_dot_types_dot_base__pb2._TASKORGANIZATION
_TASKPB.fields_by_name['algo_supplier'].message_type = lib_dot_types_dot_base__pb2._TASKORGANIZATION
_TASKPB.fields_by_name['data_suppliers'].message_type = lib_dot_types_dot_base__pb2._TASKORGANIZATION
_TASKPB.fields_by_name['power_suppliers'].message_type = lib_dot_types_dot_base__pb2._TASKORGANIZATION
_TASKPB.fields_by_name['receivers'].message_type = lib_dot_types_dot_base__pb2._TASKORGANIZATION
_TASKPB.fields_by_name['operation_cost'].message_type = lib_dot_types_dot_base__pb2._TASKRESOURCECOSTDECLARE
_TASKPB.fields_by_name['power_resource_options'].message_type = _TASKPOWERRESOURCEOPTION
_TASKPB.fields_by_name['state'].enum_type = lib_dot_types_dot_base__pb2._TASKSTATE
_TASKPB.fields_by_name['task_events'].message_type = _TASKEVENT
_TASKPOWERRESOURCEOPTION.fields_by_name['resource_used_overview'].message_type = lib_dot_types_dot_resourcedata__pb2._RESOURCEUSAGEOVERVIEW
DESCRIPTOR.message_types_by_name['TaskPB'] = _TASKPB
DESCRIPTOR.message_types_by_name['TaskPowerResourceOption'] = _TASKPOWERRESOURCEOPTION
DESCRIPTOR.message_types_by_name['TaskEvent'] = _TASKEVENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TaskPB = _reflection.GeneratedProtocolMessageType('TaskPB', (_message.Message,), {
  'DESCRIPTOR' : _TASKPB,
  '__module__' : 'lib.types.taskdata_pb2'
  # @@protoc_insertion_point(class_scope:types.TaskPB)
  })
_sym_db.RegisterMessage(TaskPB)

TaskPowerResourceOption = _reflection.GeneratedProtocolMessageType('TaskPowerResourceOption', (_message.Message,), {
  'DESCRIPTOR' : _TASKPOWERRESOURCEOPTION,
  '__module__' : 'lib.types.taskdata_pb2'
  # @@protoc_insertion_point(class_scope:types.TaskPowerResourceOption)
  })
_sym_db.RegisterMessage(TaskPowerResourceOption)

TaskEvent = _reflection.GeneratedProtocolMessageType('TaskEvent', (_message.Message,), {
  'DESCRIPTOR' : _TASKEVENT,
  '__module__' : 'lib.types.taskdata_pb2'
  # @@protoc_insertion_point(class_scope:types.TaskEvent)
  })
_sym_db.RegisterMessage(TaskEvent)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
