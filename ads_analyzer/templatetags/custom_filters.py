from django import template
import locale

register = template.Library()

@register.filter
def idr_currency(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return value
    
    # Format with thousand separator as dot, decimal as comma
    # We use basic string formatting first
    formatted = "{:,.2f}".format(value)
    
    # Swap standard US format (1,234.56) to IDR format (1.234,56)
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Prepare integer part and decimal part handling
    if formatted.endswith(",00"):
        formatted = formatted[:-3]
    
    return formatted

@register.filter
def intdot(value):
    """
    Formats an integer with dot as thousand separator (Indonesian style).
    Example: 140723133 -> 140.723.133
    """
    try:
        value = int(value)
    except (ValueError, TypeError):
        return value
    
    # Format with comma separator first: 140,723,133
    formatted = "{:,}".format(value)
    
    # Replace comma with dot: 140.723.133
    return formatted.replace(",", ".")
