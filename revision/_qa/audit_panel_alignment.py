from pathlib import Path
import json

def require_matplotlib_panel_alignment(fig,json_out=None,overlay_svg=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,exclude_axes=None):
    excluded=exclude_axes or []
    axes=[x for x in fig.axes if x not in excluded and x.get_visible()]
    width,height=fig.get_size_inches()*72
    boxes=[dict(panel=x.get_gid() or str(i),left=x.get_position().x0*width,right=x.get_position().x1*width,bottom=x.get_position().y0*height,top=x.get_position().y1*height) for i,x in enumerate(axes)]
    failures=[]
    for i,x in enumerate(boxes):
        for y in boxes[i+1:]:
            same_row=abs((x['top']+x['bottom'])/2-(y['top']+y['bottom'])/2)<tolerance_pt*4
            same_col=abs((x['left']+x['right'])/2-(y['left']+y['right'])/2)<tolerance_pt*4
            for key in (['top','bottom'] if same_row else [])+(['left','right'] if same_col else []):
                if abs(x[key]-y[key])>tolerance_pt:failures.append([x['panel'],y['panel'],key])
    result=dict(boxes=boxes,tolerance_pt=tolerance_pt,failures=failures,passed=not failures,scope='portable axes-edge check; original full QA report accompanies release')
    if json_out:Path(json_out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if overlay_svg:
        shapes=''.join('<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="red"/>'%(x['left'],height-x['top'],x['right']-x['left'],x['top']-x['bottom']) for x in boxes)
        Path(overlay_svg).write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %s %s">%s</svg>'%(width,height,shapes),encoding='utf-8')
    if strict and failures:raise ValueError('Axes alignment failed: '+str(failures))
    return result
