def title_html(text):
    return '<h3 class="component_title">{}</h3>'.format(text)


def has_single_component(component):
    from .contents import HTMLComponentGroup

    content = component.content
    if isinstance(content, HTMLComponentGroup):
        if len(content.components) == 1 and has_single_component(content.components[0]):
            return True
        else:
            return False
    else:
        return True
