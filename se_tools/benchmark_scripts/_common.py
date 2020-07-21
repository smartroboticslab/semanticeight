import datetime

def time():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

def flow_style_list(l):  # concert list into flow-style (default is block style)
    from ruamel.yaml.comments import CommentedSeq
    cs = CommentedSeq(l)
    cs.fa.set_flow_style()
    return cs
