from langchain_core.tools import tool


def create_tools(method_store):
    @tool
    def search_relevant_code(class_name: str, method_name: str = '') -> str:
        """Look up a class or method in the project AST.

        Args:
          class_name: Required. The class to search in.
          method_name: Optional. If given, return only that method; otherwise return all methods of the class.

        Examples:
          search_relevant_code(class_name="HttpResponseStatus")
            → all methods/constructors of that class, plus fields and inheritance

          search_relevant_code(class_name="ListenableFuture", method_name="cancel")
            → only ListenableFuture.cancel(), resolved through inheritance

        Use when you see an unfamiliar class or method call in the code."""
        if not class_name:
            return "Error: class_name is required."
        results = method_store.search(
            class_name=class_name,
            method_name=method_name or None,
            max_results=10,
        )

        parts = []

        if class_name and not method_name:
            ci = method_store.get_class_info(class_name)
            if ci:
                header_parts = [f"class {ci.name}"]
                if ci.extends:
                    header_parts.append(f"extends {', '.join(ci.extends)}")
                if ci.implements:
                    header_parts.append(f"implements {', '.join(ci.implements)}")
                parts.append(' '.join(header_parts))
                if ci.fields:
                    fields_str = '\n'.join(
                        f"  {ci.field_modifiers.get(fname, 'package')} {ftype} {fname};"
                        for fname, ftype in ci.fields.items()
                    )
                    parts.append(f"Fields:\n{fields_str}")

        if not results:
            if parts:
                return '\n'.join(parts)
            query = f"{class_name}.{method_name}" if class_name and method_name else (class_name or method_name)
            return f"No results found for '{query}'. This class is likely from the JDK or an external library — its source is not in this project. Do NOT search for it again."

        for mi in results:
            parts.append(mi.format())

        return '\n---\n'.join(parts)

    return [search_relevant_code]
