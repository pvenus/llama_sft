def _param_items_from_func(func):
    """Extract (param_name, details) pairs from a function spec dict."""
    args = func.get("args", [])
    if not args:
        return []
    items = []
    for arg in args:
        pname = arg.get("name", "param")
        items.append((pname, arg))
    return items

def build_functions_section(functions):
    lines = []
    # Global output format
    lines.append("Output function-call format (fixed):")
    lines.append("  <function_NAME>(param=value, ...)<end>")
    lines.append("  # Return ONLY the function call above. No extra text.")
    lines.append("")
    lines.append("Available functions:")
    for func in functions:
        description = func.get("description")
        if description:
            lines.append(f"- {description}")
        else:
            lines.append("-")
        param_items = _param_items_from_func(func)
        if param_items:
            params_str = ", ".join(f"{pname}: ..." for pname, _ in param_items)
            example = f"example function_{func.get('name', 'unknown')}({params_str})"
        else:
            example = f"example function_{func.get('name', 'unknown')}()"
        lines.append(f"  {example}")
    return "\n".join(lines)
