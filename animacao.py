import numpy as np
import plotly.graph_objects as go

def criar_retangulo_carro(x_py, y_py, w_py, h_py, scale=0.025):
    """Converte coordenadas em pixels do modelo do seguidor para unidades espaciais em metros."""
    cx = x_py + w_py / 2.0
    cy = y_py + h_py / 2.0
    rx = (cx - 25) * scale
    ry = (15 - cy) * scale 
    w = w_py * scale
    h = h_py * scale

    x_coords = [rx - w/2, rx + w/2, rx + w/2, rx - w/2, rx - w/2]
    y_coords = [ry + h/2, ry + h/2, ry - h/2, ry - h/2, ry + h/2]
    return np.array(x_coords), np.array(y_coords)

def transformar_carro(x_base, y_base, x_carro, y_carro, angulo_rad):
    """Aplica a matriz de rotação 2D e translação sobre o chassi e peças do seguidor."""
    x_b = np.array(x_base, dtype=float)
    y_b = np.array(y_base, dtype=float)
    
    x_rot = x_b * np.cos(angulo_rad) - y_b * np.sin(angulo_rad)
    y_rot = x_b * np.sin(angulo_rad) + y_b * np.cos(angulo_rad)
    
    return (x_rot + x_carro).tolist(), (y_rot + y_carro).tolist()

def criar_animacao_classica(deslocamento_x, referencia_pista, x_anim, y_anim_com_comp, y_anim_sem_comp, usar_compensador, aplicar_empurrao, t_empurrao, forca_empurrao, aplicar_constante, t_constante, velocidade_frente):
    """Gera o gráfico animado simplificado comparando as trajetórias de forma direta."""
    fig = go.Figure()
    vis_com = True if usar_compensador else False
    vis_sem = False if usar_compensador else True

    fig.add_trace(go.Scatter(x=deslocamento_x, y=referencia_pista, mode='lines', line=dict(dash='dash', color='blue', width=3), name='Centro da Pista'))
    fig.add_trace(go.Scatter(x=[x_anim[0]], y=[y_anim_com_comp[0]], mode='markers', marker=dict(symbol='square', size=16, color='green'), name='Com Compensador', visible=vis_com))
    fig.add_trace(go.Scatter(x=[x_anim[0]], y=[y_anim_com_comp[0]], mode='lines', line=dict(color='green', width=2), showlegend=False, visible=vis_com))
    fig.add_trace(go.Scatter(x=[x_anim[0]], y=[y_anim_sem_comp[0]], mode='markers', marker=dict(symbol='square', size=16, color='red'), name='Sem Compensador', visible=vis_sem))
    fig.add_trace(go.Scatter(x=[x_anim[0]], y=[y_anim_sem_comp[0]], mode='lines', line=dict(color='red', width=2), showlegend=False, visible=vis_sem))

    if aplicar_empurrao:
        x_empurrao = velocidade_frente * t_empurrao
        y_empurrao_idx = np.searchsorted(deslocamento_x, x_empurrao)
        y_pos = referencia_pista[y_empurrao_idx]
        direcao_seta = 1 if forca_empurrao > 0 else -1
        fig.add_annotation(
            x=x_empurrao, y=y_pos + (1.5 * direcao_seta), ax=x_empurrao, ay=y_pos + (0.5 * direcao_seta),
            xref="x", yref="y", axref="x", ayref="y", text="Perturbação", showarrow=True, arrowhead=2, 
            arrowsize=1.5, arrowwidth=3, arrowcolor="white", font=dict(color="white", size=14)
        )

    if aplicar_constante:
        x_start = velocidade_frente * t_constante[0]
        x_end = velocidade_frente * t_constante[1]
        fig.add_vrect(
            x0=x_start, x1=x_end, fillcolor="orange", opacity=0.15, layer="below", line_width=0,
            annotation_text="Vento Lateral Constante", annotation_position="top left", annotation_font=dict(color="orange")
        )

    frames = []
    for i in range(len(x_anim)):
        frame_data = [
            go.Scatter(x=deslocamento_x, y=referencia_pista),         
            go.Scatter(x=[x_anim[i]], y=[y_anim_com_comp[i]]),                
            go.Scatter(x=x_anim[:i+1], y=y_anim_com_comp[:i+1]),   
            go.Scatter(x=[x_anim[i]], y=[y_anim_sem_comp[i]]), 
            go.Scatter(x=x_anim[:i+1], y=y_anim_sem_comp[:i+1])    
        ]
        frames.append(go.Frame(data=frame_data, name=f"f{i}"))
            
    fig.frames = frames

    botoes_menu = [
        dict(label="Iniciar", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]),
        dict(label="Pausar", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
        dict(label="Comparação", method="update", args=[{"visible": [True, True, True, True, True]}], args2=[{"visible": [True, vis_com, vis_com, vis_sem, vis_sem]}])
    ]

    fig.update_layout(
        xaxis=dict(range=[0, max(deslocamento_x)], title="Deslocamento Longitudinal (m)"),
        yaxis=dict(range=[-4, 4], title="Deslocamento Lateral (m)"),
        height=450, margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[dict(type="buttons", direction="left", showactive=True, x=0.0, y=1.15, xanchor="left", buttons=botoes_menu)]
    )
    return fig

def criar_animacao_carro(deslocamento_x, referencia_pista, x_anim, y_anim, angulos_plotly, aplicar_empurrao, t_empurrao, forca_empurrao, aplicar_constante, t_constante, velocidade_frente):
    """Gera o ambiente visual simulando a estrada e o desenho das partes do carro (Skin)."""
    fig = go.Figure()
    largura_pista = 1.2
    pista_sup = referencia_pista + largura_pista
    pista_inf = referencia_pista - largura_pista

    fig.add_trace(go.Scatter(x=deslocamento_x, y=pista_sup, mode='lines', line=dict(color='dimgray', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=deslocamento_x, y=pista_inf, mode='lines', fill='tonexty', fillcolor='rgba(105, 105, 105, 0.8)', line=dict(color='dimgray', width=2), name='Estrada'))
    fig.add_trace(go.Scatter(x=deslocamento_x, y=referencia_pista, mode='lines', line=dict(dash='dash', color='white', width=3), name='Faixa Central'))
    fig.add_trace(go.Scatter(x=x_anim, y=y_anim, mode='lines', line=dict(color='rgba(0, 0, 0, 0.3)', width=2), name='Rastro'))

    pecas_carro = [
        ([(6,0,10,4), (34,0,10,4), (6,26,10,4), (34,26,10,4)], '#141414', 'Pneus'),
        ([(0,6,4,18)], '#323232', 'Para-choque'),
        ([(2,4,46,22)], '#DC2828', 'Chassi'),
        ([(12,6,22,18)], '#A01414', 'Cabine'),
        ([(28,7,5,16), (13,8,4,14)], '#64DCFF', 'Vidros'),
        ([(46,6,4,4), (46,20,4,4)], '#FFFF64', 'Faróis')
    ]

    base_shapes = []
    for rects, cor, nome in pecas_carro:
        x_comb, y_comb = [], []
        for r in rects:
            x_arr, y_arr = criar_retangulo_carro(*r)
            x_comb.extend(x_arr.tolist() + [np.nan])
            y_comb.extend(y_arr.tolist() + [np.nan])
        base_shapes.append((np.array(x_comb), np.array(y_comb), cor, nome))

    angulo_inicial_rad = -np.radians(angulos_plotly[0])
    for x_base, y_base, cor, nome in base_shapes:
        x_trans, y_trans = transformar_carro(x_base, y_base, x_anim[0], y_anim[0], angulo_inicial_rad)
        fig.add_trace(go.Scatter(x=x_trans, y=y_trans, fill='toself', fillcolor=cor, mode='lines', line=dict(color='black', width=1), name=nome, hoverinfo='skip'))

    if aplicar_empurrao:
        x_empurrao = velocidade_frente * t_empurrao
        y_empurrao_idx = np.searchsorted(deslocamento_x, x_empurrao)
        y_pos = referencia_pista[y_empurrao_idx]
        direcao_seta = 1 if forca_empurrao > 0 else -1
        fig.add_annotation(
            x=x_empurrao, y=y_pos + (1.5 * direcao_seta), ax=x_empurrao, ay=y_pos + (0.5 * direcao_seta),
            xref="x", yref="y", axref="x", ayref="y", text="Perturbação", showarrow=True, arrowhead=2, 
            arrowsize=1.5, arrowwidth=3, arrowcolor="white", font=dict(color="white", size=16)
        )

    if aplicar_constante:
        x_start = velocidade_frente * t_constante[0]
        x_end = velocidade_frente * t_constante[1]
        fig.add_vrect(
            x0=x_start, x1=x_end, fillcolor="orange", opacity=0.2, layer="below", line_width=0,
            annotation_text="Vento Lateral Constante", annotation_position="top left", annotation_font=dict(color="orange")
        )

    frames = []
    for i in range(len(x_anim)):
        angulo_rad = -np.radians(angulos_plotly[i])
        frame_data = [
            go.Scatter(x=deslocamento_x, y=pista_sup), 
            go.Scatter(x=deslocamento_x, y=pista_inf), 
            go.Scatter(x=deslocamento_x, y=referencia_pista), 
            go.Scatter(x=x_anim[:i+1], y=y_anim[:i+1]) 
        ]
        for x_base, y_base, cor, nome in base_shapes:
            x_trans, y_trans = transformar_carro(x_base, y_base, x_anim[i], y_anim[i], angulo_rad)
            frame_data.append(go.Scatter(x=x_trans, y=y_trans))
        frames.append(go.Frame(data=frame_data, name=f"car_f{i}"))
            
    fig.frames = frames

    fig.update_layout(
        xaxis=dict(range=[0, max(deslocamento_x)], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False, visible=False), 
        height=450, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, plot_bgcolor='#509e4a', 
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=0.0, y=1.15, xanchor="left",
            buttons=[
                dict(label="Iniciar", method="animate", args=[None, {"frame": {"duration": 40, "redraw": False}, "fromcurrent": True}]),
                dict(label="Pausar", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )
    return fig
