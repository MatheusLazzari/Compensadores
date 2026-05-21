import numpy as np
import scipy.signal as signal

class RoboPlanta:
    """
    Representa a dinâmica física do seguidor de linha.
    Modela o comportamento de um sistema de segunda ordem sem termo independente na base.
    """
    def __init__(self, m: float, b: float, K: float):
        self.m = m  # Massa do carrinho
        self.b = b  # Atrito viscoso
        self.K = K  # Ganho estático do motor/atuador

    @property
    def tf_coefs(self):
        """Retorna os coeficientes do numerador e denominador: G(s) = K / (m*s^2 + b*s)"""
        num_p = [self.K]
        den_p = [self.m, self.b, 0]  
        return num_p, den_p


class CompensadorAvancoAtraso:
    """
    Controlador Avanço-Atraso (Lead-Lag) com chaves de ativação individuais.
    Age moldando a resposta em frequência e a estabilidade transitória do sistema.
    """
    def __init__(self, ativo_av: bool = True, ativo_at: bool = True, Kc: float = 1.0, 
                 z_av: float = 1.0, p_av: float = 1.0, z_at: float = 1.0, p_at: float = 1.0):
        self.ativo_av = ativo_av
        self.ativo_at = ativo_at
        self.Kc = Kc
        self.z_av = z_av
        self.p_av = p_av
        self.z_at = z_at
        self.p_at = p_at

    @property
    def tf_coefs(self):
        """Calcula por convolução os coeficientes finais da Transfer Function do Controlador C(s)"""
        # Malha de Avanço (Lead): Se desligada, assume comportamento neutro (1.0)
        num_av = [1, self.z_av] if self.ativo_av else [1.0]
        den_av = [1, self.p_av] if self.ativo_av else [1.0]
        
        # Malha de Atraso (Lag): Se desligada, assume comportamento neutro (1.0)
        num_at = [1, self.z_at] if self.ativo_at else [1.0]
        den_at = [1, self.p_at] if self.ativo_at else [1.0]
        
        # CORREÇÃO: Se ambas as malhas estiverem desligadas, ignora o Kc (ganho = 1.0)
        ganho_efetivo = self.Kc if (self.ativo_av or self.ativo_at) else 1.0
        
        # Convolução polinomial combinando as duas partes sob o ganho efetivo
        num_c = ganho_efetivo * np.convolve(num_av, num_at)
        den_c = np.convolve(den_av, den_at)
        return num_c, den_c


class SimuladorControle:
    """
    Gerencia o laço fechado. Une a Planta e o Controlador, extraindo 
    as respostas temporais e o comportamento sob distúrbios (empurrão).
    """
    def __init__(self, planta: RoboPlanta, compensador: CompensadorAvancoAtraso):
        self.planta = planta
        self.compensador = compensador
        self._calcular_malhas()

    def _calcular_malhas(self):
        """Calcula as funções de transferência equivalentes para malha aberta, fechada e distúrbio."""
        num_p, den_p = self.planta.tf_coefs
        num_c, den_c = self.compensador.tf_coefs

        # Malha Aberta: L(s) = C(s) * G(s)
        self.num_L = np.convolve(num_c, num_p)
        self.den_L = np.convolve(den_c, den_p)
        self.sys_ma = signal.TransferFunction(self.num_L, self.den_L)

        # Malha Fechada: T(s) = L(s) / (1 + L(s))
        pad_len = max(len(self.num_L), len(self.den_L))
        num_L_padded = np.pad(self.num_L, (pad_len - len(self.num_L), 0), 'constant')
        den_L_padded = np.pad(self.den_L, (pad_len - len(self.den_L), 0), 'constant')
        self.den_T = num_L_padded + den_L_padded
        self.sys_mf = signal.TransferFunction(self.num_L, self.den_T)

        # Rejeição de Distúrbio: Td(s) = G(s) / (1 + C(s)G(s))
        num_Td = np.convolve(num_p, den_c)
        self.sys_disturbio = signal.TransferFunction(num_Td, self.den_T)

    def simular_resposta_degrau(self):
        """Simula a curva clássica de resposta ao degrau unitário com tempo dinâmico."""
        # A omissão do vetor 'T' faz o scipy.signal calcular o tempo ideal automaticamente
        t_step, y_step = signal.step(self.sys_mf)
        return t_step, y_step

    def simular_esforco_controle(self):
        """Mapeia a tensão/sinal de controle U(s) enviado para os atuadores com tempo dinâmico."""
        num_c, _ = self.compensador.tf_coefs
        _, den_p = self.planta.tf_coefs
        num_Tu = np.convolve(num_c, den_p)
        sys_esforco = signal.TransferFunction(num_Tu, self.den_T)
        
        # A omissão do vetor 'T' permite a escala automática do tempo
        t_step, y_step = signal.step(sys_esforco)
        return t_step, y_step

    def simular_pista(self, t_sim, aplicar_empurrao, t_empurrao, forca_empurrao):
        """Gera o cenário de pista sinoidal integrado à perturbação externa pontual."""
        referencia_pista = 1.5 * np.sin(0.5 * t_sim)
        sinal_disturbio = np.zeros_like(t_sim)
        
        if aplicar_empurrao:
            idx_start = np.searchsorted(t_sim, t_empurrao)
            idx_end = np.searchsorted(t_sim, t_empurrao + 0.3)  # Duração fixa do pulso (0.3s)
            sinal_disturbio[idx_start:idx_end] = forca_empurrao

        _, y_rastreamento, _ = signal.lsim(self.sys_mf, referencia_pista, t_sim)
        _, y_rejeicao, _ = signal.lsim(self.sys_disturbio, sinal_disturbio, t_sim)
        
        return referencia_pista, y_rastreamento + y_rejeicao


def calcular_lgr_continuo(num, den, K_vals):
    """
    Rastreia numericamente a evolução das raízes da equação característica.
    Garante o ordenamento correto dos ramos do LGR para evitar quebras visuais nas linhas.
    """
    pad_len = max(len(num), len(den))
    num_pad = np.pad(num, (pad_len - len(num), 0), 'constant')
    den_pad = np.pad(den, (pad_len - len(den), 0), 'constant')
    
    n_ramos = pad_len - 1
    ramos = np.zeros((len(K_vals), n_ramos), dtype=complex)
    raizes_anteriores = np.roots(den_pad)
    
    for i, k in enumerate(K_vals):
        eq_char = den_pad + k * num_pad
        raizes_atuais = np.roots(eq_char)
        
        # Algoritmo de proximidade geométrica para evitar saltos cruzados de ramos no gráfico
        if i > 0:
            raizes_ordenadas = []
            raizes_disponiveis = list(raizes_atuais)
            for raiz_antiga in raizes_anteriores:
                distancias = [np.abs(r - raiz_antiga) for r in raizes_disponiveis]
                idx_mais_proxima = np.argmin(distancias)
                raizes_ordenadas.append(raizes_disponiveis.pop(idx_mais_proxima))
            raizes_atuais = np.array(raizes_ordenadas)
            
        ramos[i, :] = raizes_atuais
        raizes_anteriores = raizes_atuais
        
    return ramos