import streamlit as st
import numpy as np
from modelagem import RoboPlanta, CompensadorAvancoAtraso, SimuladorControle
from visual import renderizar_aba_principal, renderizar_aba_calculo, renderizar_aba_guia

# =====================================================================
# CONFIGURAÇÃO E INICIALIZAÇÃO DE ESTADO DO FRAMEWORK
# =====================================================================
st.set_page_config(page_title="Seguidor de Linha", layout="wide")

# Mapeamento e persistência das variáveis de controle de sessão
valores_padrao = {
    "ativar_avanco": True, "ativar_atraso": True, "m": 1.0, "b": 1.0, "K": 1.0,
    "Kc": 25.0, "z_av": 1.0, "p_av": 10.0, "z_at": 0.5, "p_at": 0.05,
    "t_empurrao": 5.0, "forca_empurrao": 10.0,
    "t_constante": (8.0, 14.0), "forca_constante": 5.0,
    "usar_escrita": False,
    "aplicar_empurrao": False,
    "aplicar_constante": False,
    "mostrar_marcadores": False
}
for chave, val in valores_padrao.items():
    if chave not in st.session_state:
        st.session_state[chave] = val

st.title("Controle do Seguidor de Linha")

# =====================================================================
# CONSTRUÇÃO DA INTERFACE DA SIDEBAR (PAINEL LATERAL)
# =====================================================================
st.sidebar.header("Configurações de Entrada")
usar_escrita = st.sidebar.toggle("Inserir valores manualmente", key="usar_escrita")

st.sidebar.markdown("---")
st.sidebar.header("Parâmetros da Planta")
st.sidebar.latex(r"G(s) = \frac{K}{m\cdot s^2 + b\cdot s}")

if usar_escrita:
    m = st.sidebar.number_input("Massa (m)", min_value=0.1, max_value=1000.0, value=float(st.session_state["m"]), step=1.0)
    b = st.sidebar.number_input("Atrito Viscoso (b)", min_value=0.0, max_value=1000.0, value=float(st.session_state["b"]), step=1.0)
    K = st.sidebar.number_input("Ganho Estático (K)", min_value=0.1, max_value=1000.0, value=float(st.session_state["K"]), step=1.0)
else:
    m = st.sidebar.slider("Massa (m)", min_value=0.1, max_value=1000.0, value=float(st.session_state["m"]), step=0.5)
    b = st.sidebar.slider("Atrito Viscoso (b)", min_value=0.0, max_value=1000.0, value=float(st.session_state["b"]), step=0.5)
    K = st.sidebar.slider("Ganho Estático (K)", min_value=0.1, max_value=1000.0, value=float(st.session_state["K"]), step=0.5)

st.session_state["m"], st.session_state["b"], st.session_state["K"] = m, b, K

st.sidebar.header("Compensador Avanço-Atraso")

if st.sidebar.button("Calcular compensador ótimo", use_container_width=True):
    polo_lento = b / m if m > 0 else 0.1
    st.session_state["ativar_avanco"] = True
    st.session_state["ativar_atraso"] = True
    st.session_state["z_av"] = float(np.clip(polo_lento, 0.1, 50.0))
    st.session_state["p_av"] = float(np.clip(max(10.0, 5.0 * polo_lento), 0.1, 300.0))
    st.session_state["Kc"] = float(np.clip((m * (st.session_state["p_av"] ** 2)) / (2 * K), 0.1, 5000.0))
    wn_aprox = st.session_state["p_av"] / 1.414 
    st.session_state["z_at"] = float(np.clip(wn_aprox / 10.0, 0.01, 25.0))
    st.session_state["p_at"] = float(np.clip((wn_aprox / 10.0) / 10.0, 0.001, 2.5))
    st.toast("Compensador Avanço-Atraso ótimo encontrado.")
    st.rerun()

ativar_avanco = st.sidebar.checkbox("Ativar Malha de Avanço", key="ativar_avanco")
ativar_atraso = st.sidebar.checkbox("Ativar Malha de Atraso", key="ativar_atraso")

compensadores_ativos = ativar_avanco or ativar_atraso

if usar_escrita:
    Kc = st.sidebar.number_input("Ganho (Kc)", min_value=0.1, max_value=5000.0, value=float(st.session_state["Kc"]), step=0.5, disabled=not compensadores_ativos)
else:
    Kc = st.sidebar.slider("Ganho (Kc)", min_value=0.1, max_value=5000.0, value=float(st.session_state["Kc"]), step=0.5, disabled=not compensadores_ativos)

st.session_state["Kc"] = Kc

z_av, p_av = float(st.session_state["z_av"]), float(st.session_state["p_av"])
if ac := ativar_avanco:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parâmetros de Avanço")
    z_av = st.sidebar.number_input("Zero de Avanço", min_value=0.1, max_value=50.0, value=z_av) if usar_escrita else st.sidebar.slider("Zero de Avanço", min_value=0.1, max_value=50.0, value=z_av)
    p_av = st.sidebar.number_input("Polo de Avanço", min_value=0.1, max_value=300.0, value=p_av) if usar_escrita else st.sidebar.slider("Polo de Avanço", min_value=0.1, max_value=300.0, value=p_av)
    st.session_state["z_av"], st.session_state["p_av"] = z_av, p_av

z_at, p_at = float(st.session_state["z_at"]), float(st.session_state["p_at"])
if at := ativar_atraso:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Parâmetros de Atraso")
    z_at = st.sidebar.number_input("Zero de Atraso", min_value=0.01, max_value=25.0, value=z_at) if usar_escrita else st.sidebar.slider("Zero de Atraso", min_value=0.01, max_value=25.0, value=z_at)
    p_at = st.sidebar.number_input("Polo de Atraso", min_value=0.001, max_value=2.5, value=p_at) if usar_escrita else st.sidebar.slider("Polo de Atraso", min_value=0.001, max_value=2.5, value=p_at)
    st.session_state["z_at"], st.session_state["p_at"] = z_at, p_at

st.sidebar.markdown("---")
if ac and at: 
    st.sidebar.latex(r"C(s) = K_c \frac{(s+z_{av})(s+z_{at})}{(s+p_{av})(s+p_{at})}")
elif ac: 
    st.sidebar.latex(r"C(s) = K_c \frac{s+z_{av}}{s+p_{av}}")
elif at: 
    st.sidebar.latex(r"C(s) = K_c \frac{s+z_{at}}{s+p_{at}}")
else:
    st.sidebar.latex(r"C(s) = 1")
    st.sidebar.info("Sistema operando sem compensação (Malha original).")

st.sidebar.header("Perturbações Externas")

aplicar_empurrao = st.sidebar.checkbox("Ativar Empurrão Rápido", key="aplicar_empurrao")
if aplicar_empurrao:
    t_empurrao = st.sidebar.number_input("Instante do Empurrão (s)", min_value=2.0, max_value=18.0, value=float(st.session_state["t_empurrao"])) if usar_escrita else st.sidebar.slider("Instante do Empurrão (s)", min_value=2.0, max_value=18.0, value=float(st.session_state["t_empurrao"]))
    forca_empurrao = st.sidebar.number_input("Força do Empurrão", min_value=-1000.0, max_value=1000.0, value=float(st.session_state["forca_empurrao"]), step=1.0) if usar_escrita else st.sidebar.slider("Força do Empurrão", min_value=-1000.0, max_value=1000.0, value=float(st.session_state["forca_empurrao"]), step=1.0)
    st.session_state["t_empurrao"], st.session_state["forca_empurrao"] = t_empurrao, forca_empurrao
else:
    t_empurrao, forca_empurrao = st.session_state["t_empurrao"], st.session_state["forca_empurrao"]

aplicar_constante = st.sidebar.checkbox("Ativar Força Contínua (Vento/Declive)", key="aplicar_constante")
if aplicar_constante:
    if usar_escrita:
        colA, colB = st.sidebar.columns(2)
        tc_start = colA.number_input("Início (s)", min_value=0.0, max_value=20.0, value=float(st.session_state["t_constante"][0]))
        tc_end = colB.number_input("Fim (s)", min_value=0.0, max_value=20.0, value=float(st.session_state["t_constante"][1]))
        t_constante = (tc_start, tc_end)
    else:
        t_constante = st.sidebar.slider("Intervalo de Duração (s)", min_value=0.0, max_value=20.0, value=st.session_state["t_constante"])
        
    forca_constante = st.sidebar.number_input("Força Contínua", min_value=-1000.0, max_value=1000.0, value=float(st.session_state["forca_constante"]), step=1.0) if usar_escrita else st.sidebar.slider("Força Contínua", min_value=-1000.0, max_value=1000.0, value=float(st.session_state["forca_constante"]), step=1.0)
    st.session_state["t_constante"], st.session_state["forca_constante"] = t_constante, forca_constante
    
    st.sidebar.warning("Aviso: O retorno lento para estabilidade acontece porque a perturbação enxerga o polo que foi cancelado matematicamente pelo controlador!")
    with st.sidebar.expander("O que é esse cancelamento?"):
        st.write("O controlador Avanço-Atraso anula a inércia lenta do carrinho inserindo um 'zero' em cima do 'polo' original. Isso deixa a resposta à referência muito rápida.")
        st.write("No entanto, o vento atinge o carrinho fisicamente *depois* do controlador. A dinâmica lenta original da planta ainda está lá e, como o controlador foi sintonizado para 'ignorá-la', ele demora a rejeitar a perturbação externa.")
else:
    t_constante, forca_constante = st.session_state["t_constante"], st.session_state["forca_constante"]

planta = RoboPlanta(m, b, K)
compensador = CompensadorAvancoAtraso(ativo_av=ativar_avanco, ativo_at=ativar_atraso, Kc=Kc, z_av=z_av, p_av=p_av, z_at=z_at, p_at=p_at)
simulador = SimuladorControle(planta, compensador)

t_step, y_step = simulador.simular_resposta_degrau()

# =====================================================================
# GERENCIAMENTO E RENDERIZAÇÃO DO LAYOUT DE ABAS MAIN
# =====================================================================
tab_principal, tab_calculo, tab_guia = st.tabs(["Principal", "Compensador ótimo", "Guia"])

with tab_principal:
    renderizar_aba_principal(simulador, t_step, y_step, aplicar_empurrao, t_empurrao, forca_empurrao, aplicar_constante, t_constante, forca_constante, ativar_avanco, ativar_atraso, planta)

with tab_calculo:
    renderizar_aba_calculo(m, b, K)

with tab_guia:
    renderizar_aba_guia()
