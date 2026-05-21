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
        
        # Variável para verificar se algum compensador está ativo
        usar_compensador = ativar_avanco or ativar_atraso

        # Sistema COM Compensador
        sys_ma_comp = ctl.tf(simulador.num_L, simulador.den_L)
        
        # Sistema SEM Compensador
        comp_desligado = CompensadorAvancoAtraso(ativo_av=False, ativo_at=False, Kc=1.0)
        sim_sem_comp = SimuladorControle(planta, comp_desligado)
        sys_ma_uncomp = ctl.tf(sim_sem_comp.num_L, sim_sem_comp.den_L)
        
        # Mapeamento dinâmico dos polos e zeros globais para dimensionar o eixo X
        raizes_globais = np.concatenate((np.roots(simulador.num_L), np.roots(simulador.den_L)))
        raizes_validas = np.abs(raizes_globais[raizes_globais != 0])
        
        if len(raizes_validas) > 0:
            w_min = np.floor(np.log10(np.min(raizes_validas))) - 1
            w_max = np.ceil(np.log10(np.max(raizes_validas))) + 1
        else:
            w_min, w_max = -2, 2
            
        w = np.logspace(w_min, w_max, 500)
        
        # ==========================================================
        # 1. DESENHO DAS LINHAS PRINCIPAIS (VERDE E/OU VERMELHA)
        # ==========================================================
        if usar_compensador:
            # Se compensador está ativo, desenha o Verde como base e adiciona o Vermelho por baixo
            fig_bode = bode(sys_ma_comp, w=w)
            if len(fig_bode.data) >= 2:
                fig_bode.data[0].line.color = 'green'
                fig_bode.data[0].name = 'Com Compensador'
                fig_bode.data[0].showlegend = True
                fig_bode.data[1].line.color = 'green'
                fig_bode.data[1].showlegend = False
                
            mag_u, phase_u, omega_u = ctl.bode(sys_ma_uncomp, w, plot=False)
            mag_u_db = 20 * np.log10(mag_u.flatten())
            phase_u_deg = np.degrees(phase_u.flatten())
            
            fig_bode.add_trace(go.Scatter(x=omega_u.flatten(), y=mag_u_db, mode='lines', line=dict(color='red'), name='Sem Compensador'), row=1, col=1)
            fig_bode.add_trace(go.Scatter(x=omega_u.flatten(), y=phase_u_deg, mode='lines', line=dict(color='red'), showlegend=False), row=2, col=1)
        else:
            # Se não há compensador, desenha APENAS a curva Vermelha
            fig_bode = bode(sys_ma_uncomp, w=w)
            if len(fig_bode.data) >= 2:
                fig_bode.data[0].line.color = 'red'
                fig_bode.data[0].name = 'Sem Compensador'
                fig_bode.data[0].showlegend = True
                fig_bode.data[1].line.color = 'red'
                fig_bode.data[1].showlegend = False


        # ==========================================================
        # 2. MARCADORES 'X' e 'O' NA LINHA DO GRÁFICO (POLOS E ZEROS)
        # ==========================================================
        
        # --- RAÍZES: COM COMPENSADOR (VERDE) ---
        if usar_compensador:
            zeros_c = np.roots(simulador.num_L)
            polos_c = np.roots(simulador.den_L)
            freq_zeros_c = np.unique(np.abs(zeros_c[zeros_c != 0]))
            freq_polos_c = np.unique(np.abs(polos_c[polos_c != 0]))

            if len(freq_polos_c) > 0:
                mag_p, phase_p, _ = ctl.bode(sys_ma_comp, freq_polos_c, plot=False)
                mag_p_db = 20 * np.log10(mag_p.flatten())
                phase_p_deg = np.degrees(phase_p.flatten())
                texto_polos = [f"Polo (Com Comp)<br>Frequência: {wp:.3f} rad/s<br>Magnitude: {mp:.1f} dB<br>Fase: {fp:.1f}°" for wp, mp, fp in zip(freq_polos_c, mag_p_db, phase_p_deg)]
                
                fig_bode.add_trace(go.Scatter(
                    x=freq_polos_c, y=mag_p_db, mode='markers',
                    marker=dict(symbol='x', color='green', size=10, line=dict(width=2)),
                    hoverinfo='text', hovertext=texto_polos, showlegend=True, name="Polos (Com Comp)"
                ), row=1, col=1)
                
                fig_bode.add_trace(go.Scatter(
                    x=freq_polos_c, y=phase_p_deg, mode='markers',
                    marker=dict(symbol='x', color='green', size=10, line=dict(width=2)),
                    hoverinfo='text', hovertext=texto_polos, showlegend=False
                ), row=2, col=1)

            if len(freq_zeros_c) > 0:
                mag_z, phase_z, _ = ctl.bode(sys_ma_comp, freq_zeros_c, plot=False)
                mag_z_db = 20 * np.log10(mag_z.flatten())
                phase_z_deg = np.degrees(phase_z.flatten())
                texto_zeros = [f"Zero (Com Comp)<br>Frequência: {wz:.3f} rad/s<br>Magnitude: {mz:.1f} dB<br>Fase: {fz:.1f}°" for wz, mz, fz in zip(freq_zeros_c, mag_z_db, phase_z_deg)]
                
                fig_bode.add_trace(go.Scatter(
                    x=freq_zeros_c, y=mag_z_db, mode='markers',
                    marker=dict(symbol='circle-open', color='green', size=10, line=dict(width=3)),
                    hoverinfo='text', hovertext=texto_zeros, showlegend=True, name="Zeros (Com Comp)"
                ), row=1, col=1)
                
                fig_bode.add_trace(go.Scatter(
                    x=freq_zeros_c, y=phase_z_deg, mode='markers',
                    marker=dict(symbol='circle-open', color='green', size=10, line=dict(width=3)),
                    hoverinfo='text', hovertext=texto_zeros, showlegend=False
                ), row=2, col=1)

        # --- RAÍZES: SEM COMPENSADOR (VERMELHO) ---
        zeros_u = np.roots(sim_sem_comp.num_L)
        polos_u = np.roots(sim_sem_comp.den_L)
        freq_zeros_u = np.unique(np.abs(zeros_u[zeros_u != 0]))
        freq_polos_u = np.unique(np.abs(polos_u[polos_u != 0]))

        if len(freq_polos_u) > 0:
            mag_p_u, phase_p_u, _ = ctl.bode(sys_ma_uncomp, freq_polos_u, plot=False)
            mag_p_u_db = 20 * np.log10(mag_p_u.flatten())
            phase_p_u_deg = np.degrees(phase_p_u.flatten())
            texto_polos_u = [f"Polo (Sem Comp)<br>Frequência: {wp:.3f} rad/s<br>Magnitude: {mp:.1f} dB<br>Fase: {fp:.1f}°" for wp, mp, fp in zip(freq_polos_u, mag_p_u_db, phase_p_u_deg)]
            
            fig_bode.add_trace(go.Scatter(
                x=freq_polos_u, y=mag_p_u_db, mode='markers',
                marker=dict(symbol='x', color='red', size=10, line=dict(width=2)),
                hoverinfo='text', hovertext=texto_polos_u, showlegend=True, name="Polos (Sem Comp)"
            ), row=1, col=1)
            
            fig_bode.add_trace(go.Scatter(
                x=freq_polos_u, y=phase_p_u_deg, mode='markers',
                marker=dict(symbol='x', color='red', size=10, line=dict(width=2)),
                hoverinfo='text', hovertext=texto_polos_u, showlegend=False
            ), row=2, col=1)

        if len(freq_zeros_u) > 0:
            mag_z_u, phase_z_u, _ = ctl.bode(sys_ma_uncomp, freq_zeros_u, plot=False)
            mag_z_u_db = 20 * np.log10(mag_z_u.flatten())
            phase_z_u_deg = np.degrees(phase_z_u.flatten())
            texto_zeros_u = [f"Zero (Sem Comp)<br>Frequência: {wz:.3f} rad/s<br>Magnitude: {mz:.1f} dB<br>Fase: {fz:.1f}°" for wz, mz, fz in zip(freq_zeros_u, mag_z_u_db, phase_z_u_deg)]
            
            fig_bode.add_trace(go.Scatter(
                x=freq_zeros_u, y=mag_z_u_db, mode='markers',
                marker=dict(symbol='circle-open', color='red', size=10, line=dict(width=3)),
                hoverinfo='text', hovertext=texto_zeros_u, showlegend=True, name="Zeros (Sem Comp)"
            ), row=1, col=1)
            
            fig_bode.add_trace(go.Scatter(
                x=freq_zeros_u, y=phase_z_u_deg, mode='markers',
                marker=dict(symbol='circle-open', color='red', size=10, line=dict(width=3)),
                hoverinfo='text', hovertext=texto_zeros_u, showlegend=False
            ), row=2, col=1)

        # ==========================================================
        # 3. FREQUÊNCIAS DE CRUZAMENTO (0 dB) E MARCADORES INTERATIVOS
        # ==========================================================
        _, pm_c, _, wp_c = ctl.margin(sys_ma_comp)
        _, pm_u, _, wp_u = ctl.margin(sys_ma_uncomp)
        
        # Lógica para o sistema Sem Compensador (Vermelho) - LOSANGO
        if not np.isnan(wp_u) and not np.isinf(wp_u) and wp_u > 0:
            fase_u = pm_u - 180.0
            texto_hover_u = f"<b>Crossover (Sem Comp)</b><br>Frequência: {wp_u:.3f} rad/s<br>Magnitude: 0.0 dB<br>Fase: {fase_u:.1f}°"
            
            fig_bode.add_trace(go.Scatter(
                x=[wp_u], y=[0], mode='markers',
                marker=dict(color='red', size=12, symbol='diamond', line=dict(color='white', width=2)),
                hoverinfo='text', hovertext=texto_hover_u, showlegend=True, name="0 dB (Sem Comp)"
            ), row=1, col=1)
            
            fig_bode.add_trace(go.Scatter(
                x=[wp_u], y=[fase_u], mode='markers',
                marker=dict(color='red', size=12, symbol='diamond', line=dict(color='white', width=2)),
                hoverinfo='text', hovertext=texto_hover_u, showlegend=False
            ), row=2, col=1)
            
        # Lógica para o sistema Com Compensador (Verde) - LOSANGO (Apenas se ativo)
        if usar_compensador and not np.isnan(wp_c) and not np.isinf(wp_c) and wp_c > 0:
            fase_c = pm_c - 180.0
            texto_hover_c = f"<b>Crossover (Com Comp)</b><br>Frequência: {wp_c:.3f} rad/s<br>Magnitude: 0.0 dB<br>Fase: {fase_c:.1f}°"
            
            fig_bode.add_trace(go.Scatter(
                x=[wp_c], y=[0], mode='markers',
                marker=dict(color='green', size=12, symbol='diamond', line=dict(color='white', width=2)),
                hoverinfo='text', hovertext=texto_hover_c, showlegend=True, name="0 dB (Com Comp)"
            ), row=1, col=1)
            
            fig_bode.add_trace(go.Scatter(
                x=[wp_c], y=[fase_c], mode='markers',
                marker=dict(color='green', size=12, symbol='diamond', line=dict(color='white', width=2)),
                hoverinfo='text', hovertext=texto_hover_c, showlegend=False
            ), row=2, col=1)

        # Configura o layout, atualiza os eixos para 10^x e posiciona a legenda estrategicamente
        fig_bode.update_xaxes(type='log', exponentformat='power', row='all', col=1)
        fig_bode.update_layout(
            height=400, 
            margin=dict(l=0, r=0, t=30, b=0), 
            legend=dict(
                orientation="h", y=-0.2, x=0.5, xanchor="center", 
                itemclick="toggle", itemdoubleclick="toggleothers"
            ),
            hovermode='closest'
        )
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
        
        zeros_ma = np.roots(simulador.num_L)
        polos_ma = np.roots(simulador.den_L)
        polos_atuais = np.roots(simulador.den_T)
        
        fig_rl.add_trace(go.Scatter(x=np.real(zeros_ma), y=np.imag(zeros_ma), mode='markers', marker=dict(symbol='circle-open', size=10, color='red', line=dict(width=2)), name='Zeros MA'))
        fig_rl.add_trace(go.Scatter(x=np.real(polos_ma), y=np.imag(polos_ma), mode='markers', marker=dict(symbol='x', size=10, color='red', line=dict(width=2)), name='Polos MA'))
        fig_rl.add_trace(go.Scatter(x=np.real(polos_atuais), y=np.imag(polos_atuais), mode='markers', marker=dict(symbol='diamond', size=12, color='green'), name='Polos Atuais'))
        
        fig_rl.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Eixo Real", yaxis_title="Eixo Imag", legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
        fig_rl.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
        fig_rl.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        
        # Calcula a "bounding box" apenas nos polos/zeros de interesse
        pontos_interesse = np.concatenate((zeros_ma, polos_ma, polos_atuais))
        if len(pontos_interesse) > 0:
            x_min, x_max = np.min(np.real(pontos_interesse)), np.max(np.real(pontos_interesse))
            y_max_abs = np.max(np.abs(np.imag(pontos_interesse)))
            
            # Adiciona pequenas margens
            margem_x = max(x_max - x_min, 1.0) * 0.4
            margem_y = max(y_max_abs, 1.0) * 1.5
            
            fig_rl.update_xaxes(range=[x_min - margem_x, x_max + margem_x])
            fig_rl.update_yaxes(range=[-(y_max_abs + margem_y), y_max_abs + margem_y], scaleanchor="x", scaleratio=1)
        else:
            fig_rl.update_yaxes(scaleanchor="x", scaleratio=1)

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
    Kc_bruto = (m * (p_av_id ** 2)) / (2 * K)
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
        st.latex(rf"p_{{av}} = 5 \cdot z_{{av}} = 5 \cdot {p_lento:.3f} = {p_av_id:.3f}")
        if 5.0 * p_lento < 10.0:
            st.caption(f"*OBS: O valor calculado foi de {5.0 * p_lento:.3f}, mas foi ajustado para 10.000 por ser o limite mínimo do projeto.*")
        st.markdown("**Passo C: Ganho para Amortecimento Ótimo**")
        st.latex(rf"K_c = \frac{{m \cdot p_{{av}}^2}}{{2 \cdot K}} = \frac{{{m} \cdot {p_av_id:.2f}^2}}{{2 \cdot {K}}} = {Kc_id:.3f}")
        if Kc_bruto > 5000.0:
            st.caption(f"*OBS: O ganho teórico calculado seria {Kc_bruto:.3f}, mas foi limitado a 5000.000 por ser o limite máximo permitido pelo sistema.*")

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
