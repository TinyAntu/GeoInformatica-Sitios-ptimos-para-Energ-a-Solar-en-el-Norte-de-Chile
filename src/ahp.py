"""AHP (Analytic Hierarchy Process) con validación por Ratio de Consistencia.

Deriva los pesos de los criterios desde una matriz de comparación por pares
(escala de Saaty 1–9) usando el autovector principal, y calcula el Ratio de
Consistencia (CR = CI / RI). Un CR < 0.10 indica que los juicios del experto son
suficientemente consistentes (Saaty, 1980); por encima de 0.10 conviene revisar
la matriz.

Esto respalda formalmente el baseline AHP del proyecto: los pesos de cada perfil
de inversión dejan de ser constantes "a mano" y pasan a derivarse de una matriz de
juicios auditada por su CR.

Referencia: Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
"""
from __future__ import annotations
import numpy as np

# Índice de Consistencia Aleatorio (Random Index) de Saaty, según el orden n de la
# matriz. Es el CI promedio de matrices recíprocas aleatorias de tamaño n.
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}


def calcular_pesos_ahp(matriz) -> dict:
    """Calcula pesos AHP y el diagnóstico de consistencia de una matriz de pares.

    Parámetros
    ----------
    matriz : array-like (n x n)
        Matriz de comparación por pares recíproca en escala de Saaty
        (A[i][j] = importancia del criterio i frente al j; A[j][i] = 1/A[i][j]).

    Devuelve
    --------
    dict con:
        pesos       : np.ndarray (n,) normalizada a suma 1 (autovector principal).
        lambda_max  : autovalor principal.
        CI          : Índice de Consistencia = (lambda_max - n) / (n - 1).
        RI          : Random Index para ese n.
        CR          : Ratio de Consistencia = CI / RI.
        consistente : bool, True si CR < 0.10.
    """
    A = np.asarray(matriz, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matriz de comparación por pares debe ser cuadrada.")
    n = A.shape[0]

    # Autovector principal (método del autovalor de Saaty).
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[idx].real)

    w = np.abs(eigvecs[:, idx].real)
    pesos = w / w.sum()

    # Consistencia.
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RANDOM_INDEX.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    return {
        'pesos': pesos,
        'lambda_max': lambda_max,
        'CI': CI,
        'RI': RI,
        'CR': CR,
        'consistente': bool(CR < 0.10),
    }


def pesos_perfil(nombre_perfil: str, ahp_config: dict) -> dict | None:
    """Devuelve {criterio: peso} para un perfil a partir de su matriz AHP en config.

    Espera un bloque de config con la forma:
        criterios: [ghi, slope, elev, northness, dist_transmision, dist_almacen, dist_subestaciones]
        <nombre_perfil>:
          matriz: [[...], [...], ...]   # n x n en escala de Saaty

    Imprime el CR y avisa si la matriz no es consistente (CR >= 0.10).
    Devuelve None si no hay matriz definida para ese perfil (para permitir fallback).
    """
    if not ahp_config:
        return None
    criterios = ahp_config.get('criterios')
    bloque = ahp_config.get(nombre_perfil)
    if not criterios or not bloque or 'matriz' not in bloque:
        return None

    res = calcular_pesos_ahp(bloque['matriz'])
    estado = "OK" if res['consistente'] else "REVISAR (CR >= 0.10)"
    print(f"  [AHP] Perfil '{nombre_perfil}': CR = {res['CR']:.4f} ({estado}) | "
          f"lambda_max = {res['lambda_max']:.3f}")
    if not res['consistente']:
        print("  [AVISO] La matriz de comparación por pares no es consistente; "
              "revisa tus juicios (deberían dar CR < 0.10).")

    return {crit: float(peso) for crit, peso in zip(criterios, res['pesos'])}
