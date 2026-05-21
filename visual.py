import streamlit as st
import numpy as np
import control as ctl
import plotly.graph_objects as go
from control_plotly import bode, step
from modelagem import calcular_lgr_continuo, CompensadorAvancoAtraso, SimuladorControle
from animacao import criar_animacao_classica, criar_animacao_carro

# =====================================================================
# SEÇÃO: COMPONENTES VISUAIS (MÉTRICAS E INDICADORES)
# =====================================================================

def renderizar_painel_analise(simulador, t_step, y_step):
    """
    Calcula e desenha o grid de cartões de métricas na tela com base
    na resposta ao degrau e margens de estabilidade.
    """
    st.subheader("Indicadores de Desempenho")

    # Tratamento matemático das variáveis de saída do Step Response
    y_final = y_step[-1]
    overshoot = max(0, (np.max(y_step) - y_final) / np.abs(y_final) * 100) if abs(y_final) > 0.001 else 0
    erro_abs = np.abs(y_step - y_final)
    dentro_faixa = erro_abs <= 0.02 * np.abs(y_final)
    indices_fora = np.where(~dentro_faixa)[0]
    ts = t_step[indices_fora[-1]] if len(indices_fora) > 0 else 0
    erro_regime = np.abs(1.0 - y_final)

    # Margens de estabilidade do sistema em Malha Aberta
    sys_ma = ctl.tf(simulador.num_L, simulador.den_L)
    gm, pm, wg, wp = ctl.margin(sys_ma)
    pm_str = "∞" if np.isinf(pm) or np.isnan(pm) else f"{pm:.1f}°"

    # Construção do grid visual de métricas
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Sobressinal (Overshoot)", f"{overshoot:.1f}%")
    col_m2.metric("Tempo de Acomodação (ts)", f"{ts:.2f} s")
    col_m3.metric("Erro em Regime (ess)", f"{erro_regime:.3f}")
    col_m4.metric("Margem de Fase (MF)", pm_str)


# =====================================================================
# SEÇÃO: RENDERIZAÇÃO DAS ABAS PRINCIPAIS
# =====================================================================

def renderizar_aba_principal(simulador, t_step, y_step, aplicar_empurrao, t_empurrao, forca_empurrao, ativar_avanco, ativar_atraso, planta):
    """Renderiza os gráficos principais de resposta temporal, frequência e simulação de pista."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resposta ao Degrau")
        sys_mf = ctl.tf(simulador.num_L, simulador.den_T)
        fig_step = step(sys_mf, t=t_step)
        fig_step.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig_step.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Tempo (s)")
        st.plotly_chart(fig_step, use_container_width=True, key="grafico_degrau")

    with col2:
        st.subheader("Diagrama de Bode (Resposta em Frequência)")
        sys_ma = ctl.tf(simulador.num_L, simulador.den_L)
        
        # Mapeamento dinâmico dos polos e zeros para escalar a janela do Bode
        raizes = np.concatenate((np.roots(simulador.num_L), np.roots(simulador.den_L)))
        raizes_validas = np.abs(raizes[raizes != 0])
        
        if len(raizes_validas) > 0:
            # Encontra a menor e a maior raiz e adiciona uma década de margem
            w_min = np.floor(np.log10(np.min(raizes_validas))) - 1
            w_max = np.ceil(np.log10(np.max(raizes_validas))) + 1
        else:
            w_min, w_max = -2, 2
            
        w = np.logspace(w_min, w_max, 500)
        
        fig_bode = bode(sys_ma, w=w)
        fig_bode.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bode, use_container_width=True, key="grafico_bode")

    st.divider()
    # Executa o painel de análise embutido localmente
    renderizar_painel_analise(simulador, t_step, y_step)
    st.divider()

    # Bloco Inferior: Esforço U(s) e Lugar das Raízes (LGR)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Esforço de Controle U(s)")
        t_esforco, y_esforco = simulador.simular_esforco_controle()
        num_c, _ = simulador.compensador.tf_coefs
        _, den_p = simulador.planta.tf_coefs
        num_Tu = np.convolve(num_c, den_p)
        sys_esforco = ctl.tf(num_Tu, simulador.den_T)
        fig_esforco = step(sys_esforco, t=t_esforco)
        fig_esforco.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Tempo (s)", yaxis_title="Sinal U(s)")
        if fig_esforco.data:
            fig_esforco.data[0].line.color = 'red'
        st.plotly_chart(fig_esforco, use_container_width=True, key="grafico_esforco_u")

    with col4:
        st.subheader("Lugar das Raízes (LGR)")
        K_vals = np.concatenate((np.linspace(0, 50, 2000), np.logspace(np.log10(50.1), 3, 1000)))
        ramos = calcular_lgr_continuo(simulador.num_L, simulador.den_L, K_vals)
        fig_rl = go.Figure()
        cores_ramos = ['#1f77b4', '#ff7f0e', '#9467bd', '#e377c2', '#17becf']
        for i in range(ramos.shape[1]):
            fig_rl.add_trace(go.Scatter(x=np.real(ramos[:, i]), y=np.imag(ramos[:, i]), mode='lines', line=dict(color=cores_ramos[i % len(cores_ramos)], width=2.5), showlegend=False))
        
        fig_rl.add_trace(go.Scatter(x=np.real(np.roots(simulador.num_L)), y=np.imag(np.roots(simulador.num_L)), mode='markers', marker=dict(symbol='circle-open', size=10, color='red', line=dict(width=2)), name='Zeros MA'))
        fig_rl.add_trace(go.Scatter(x=np.real(np.roots(simulador.den_L)), y=np.imag(np.roots(simulador.den_L)), mode='markers', marker=dict(symbol='x', size=10, color='red', line=dict(width=2)), name='Polos MA'))
        fig_rl.add_trace(go.Scatter(x=np.real(np.roots(simulador.den_T)), y=np.imag(np.roots(simulador.den_T)), mode='markers', marker=dict(symbol='diamond', size=12, color='green'), name='Polos Atuais'))
        
        fig_rl.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Eixo Real", yaxis_title="Eixo Imag", legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        fig_rl.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_rl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_rl.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        st.plotly_chart(fig_rl, use_container_width=True, key="grafico_root_locus")

    st.divider()
    
    # Simulação Dinâmica da Trajetória Real na Pista
    st.subheader("Simulação de Trajetória")
    t_sim = np.linspace(0, 20, 600)
    velocidade_frente = 1.0  
    deslocamento_x = velocidade_frente * t_sim
    referencia_pista, y_total = simulador.simular_pista(t_sim, aplicar_empurrao, t_empurrao, forca_empurrao)

    step_anim = 4
    x_anim = deslocamento_x[::step_anim]
    y_anim = y_total[::step_anim]

    comp_desligado = CompensadorAvancoAtraso(ativo_av=False, ativo_at=False, Kc=1.0)
    sim_sem_comp = SimuladorControle(planta, comp_desligado)
    _, y_total_sem_comp = sim_sem_comp.simular_pista(t_sim, aplicar_empurrao, t_empurrao, forca_empurrao)
    y_anim_sem_comp = y_total_sem_comp[::step_anim]

    usar_compensador = ativar_avanco or ativar_atraso
    y_anim_carro = y_anim if usar_compensador else y_anim_sem_comp
    angulos_plotly = -np.degrees(np.arctan2(np.gradient(y_anim_carro), np.gradient(x_anim)))

    tab_graficos, tab_carro = st.tabs(["Simulação gráfica", "Demonstração (Carro)"])
    with tab_graficos:
        fig_pista = criar_animacao_classica(deslocamento_x, referencia_pista, x_anim, y_anim, y_anim_sem_comp, usar_compensador, aplicar_empurrao, t_empurrao, forca_empurrao, velocidade_frente)
        st.plotly_chart(fig_pista, use_container_width=True, key="grafico_pista_classico")
    with tab_carro:
        fig_carro = criar_animacao_carro(deslocamento_x, referencia_pista, x_anim, y_anim_carro, angulos_plotly, aplicar_empurrao, t_empurrao, forca_empurrao, velocidade_frente)
        st.plotly_chart(fig_carro, use_container_width=True, key="grafico_skin_carro")


def renderizar_aba_calculo(m, b, K):
    """Exibe o algoritmo matemático detalhado passo a passo na tela via LaTeX."""
    st.header("Algoritmo de Sintonia Analítica Ótima")
    p_lento = b / m if m > 0 else 0.1
    p_av_id = max(10.0, 5.0 * p_lento)
    Kc_id = (m * (p_av_id ** 2)) / (2 * K)
    wn_id = p_av_id / 1.414
    z_at_id = wn_id / 10.0
    p_at_id = z_at_id / 10.0

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("1. Projeto da Malha de Avanço (Transitório)")
        st.markdown("**Passo A: Cancelamento de Polo Lento**")
        st.latex(rf"z_{{av}} = \frac{{b}}{{m}} = \frac{{{b}}}{{{m}}} = {p_lento:.3f}")
        st.markdown("**Passo B: Alocação do Polo Rápido**")
        if 5.0 * p_lento < 10.0:
            # Se caiu no piso de 10.0, explica visualmente usando a função max()
            st.latex(rf"p_{{av}} = \max(10.0, \, 5 \cdot z_{{av}}) = \max(10.0, \, {5.0 * p_lento:.3f}) = {p_av_id:.3f}")
        else:
        # Se o cálculo natural passou de 10.0, mantém a multiplicação direta original
            st.latex(rf"p_{{av}} = 5 \cdot z_{{av}} = 5 \cdot {p_lento:.3f} = {p_av_id:.3f}")
        st.markdown("**Passo C: Ganho para Amortecimento Ótimo**")
        st.latex(rf"K_c = \frac{{m \cdot p_{{av}}^2}}{{2 \cdot K}} = \frac{{{m} \cdot {p_av_id:.2f}^2}}{{2 \cdot {K}}} = {Kc_id:.3f}")

    with col_c2:
        st.subheader("2. Projeto da Malha de Atraso (Regime permanente)")
        st.markdown("**Passo D: Mapeamento de Frequência de Corte**")
        st.latex(rf"\omega_n \approx \frac{{p_{{av}}}}{{\sqrt{{2}}}} = \frac{{{p_av_id:.2f}}}{{1.414}} = {wn_id:.3f}\text{{ rad/s}}")
        st.markdown("**Passo E: Inserção do Zero de Atraso**")
        st.latex(rf"z_{{at}} = \frac{{\omega_n}}{{10}} = \frac{{{wn_id:.3f}}}{{10}} = {z_at_id:.3f}")
        st.markdown("**Passo F: Posicionamento do Polo de Atraso**")
        st.latex(rf"p_{{at}} = \frac{{z_{{at}}}}{{10}} = \frac{{{z_at_id:.3f}}}{{10}} = {p_at_id:.3f}")


def renderizar_aba_guia():
    """Gera o manual conceitual e teórico explicativo da aplicação."""
    st.header("Guia de Engenharia de Controle: Compensadores Avanço-Atraso")
    st.markdown("Em sistemas reais apenas aumentar o ganho proporcional ($K_c$) não basta. Se você aumentar demais o ganho simples, o carrinho começa a oscilar violentamente e sai da pista.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("### Malha de Avanço (Lead)")
        st.markdown("* **O que faz:** Adiciona fase positiva ao sistema próximo à frequência de cruzamento (ação semelhante à derivativa).\n* **Efeito prático:** Atua como um amortecedor preditivo, freando o seguidor antes que ele passe direto pela linha.")
    with col_g2:
        st.markdown("### Malha de Atraso (Lag)")
        st.markdown("* **O que faz:** Adiciona ganho massivo em frequências baixas (ação semelhante à integral).\n* **Efeito prático:** Anula o erro estático de regime permanente, rejeitando vento lateral ou perturbações.")
