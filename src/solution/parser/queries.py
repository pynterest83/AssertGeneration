# Tree-sitter S-expression queries per language.
# Captures: class.name/def, method.name/def, call.name/call, heritage.*, field.*

# Java

JAVA_QUERIES = """
; Classes, Interfaces, Enums
(class_declaration name: (identifier) @class.name) @class.def
(interface_declaration name: (identifier) @class.name) @class.def
(enum_declaration name: (identifier) @class.name) @class.def

; Methods & Constructors
(method_declaration name: (identifier) @method.name) @method.def
(constructor_declaration name: (identifier) @method.name) @method.def

; Method invocation calls
(method_invocation name: (identifier) @call.name) @call
(method_invocation object: (_) name: (identifier) @call.name) @call

; Constructor calls: new Foo()
(object_creation_expression type: (type_identifier) @call.name) @call

; Heritage — extends class
(class_declaration name: (identifier) @heritage.class
  (superclass (type_identifier) @heritage.extends)) @heritage

; Heritage — implements interfaces
(class_declaration name: (identifier) @heritage.class
  (super_interfaces (type_list (type_identifier) @heritage.implements))) @heritage.impl

; Field declarations
(field_declaration
  type: (_) @field.type
  declarator: (variable_declarator name: (identifier) @field.name)) @field.def
"""

# Python

PYTHON_QUERIES = """
; Classes
(class_definition name: (identifier) @class.name) @class.def

; Functions / Methods
(function_definition name: (identifier) @method.name) @method.def

; Calls
(call function: (identifier) @call.name) @call
(call function: (attribute attribute: (identifier) @call.name)) @call

; Heritage — Python class inheritance
(class_definition
  name: (identifier) @heritage.class
  superclasses: (argument_list
    (identifier) @heritage.extends)) @heritage
"""

# JavaScript

JAVASCRIPT_QUERIES = """
; Classes
(class_declaration name: (identifier) @class.name) @class.def

; Functions
(function_declaration name: (identifier) @method.name) @method.def

; Methods inside class body
(method_definition name: (property_identifier) @method.name) @method.def

; Arrow / function expression assigned to const
(lexical_declaration
  (variable_declarator
    name: (identifier) @method.name
    value: (arrow_function))) @method.def

(lexical_declaration
  (variable_declarator
    name: (identifier) @method.name
    value: (function_expression))) @method.def

; Calls
(call_expression function: (identifier) @call.name) @call
(call_expression
  function: (member_expression
    property: (property_identifier) @call.name)) @call

; Constructor calls: new Foo()
(new_expression constructor: (identifier) @call.name) @call

; Heritage — class extends
(class_declaration
  name: (identifier) @heritage.class
  (class_heritage
    (identifier) @heritage.extends)) @heritage
"""

# Mapping

LANGUAGE_QUERIES = {
    "java": JAVA_QUERIES,
    "python": PYTHON_QUERIES,
    "javascript": JAVASCRIPT_QUERIES,
}
