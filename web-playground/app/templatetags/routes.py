from django import template
register = template.Library()
    
@register.simple_tag(name='similar_words_path')
def similar_words_path():
    return 'similar_words'
    #return 'worde4mde/app/similar_words'
