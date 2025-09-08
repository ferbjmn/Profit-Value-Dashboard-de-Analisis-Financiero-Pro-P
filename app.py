# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO (SCRIPT COMPLETO)
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Parámetros editables (valores por defecto)
Rf = 0.0435   # riesgo libre
Rm = 0.085    # retorno mercado
Tc0 = 0.21    # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    """Busca la primera fila cuyo índice coincida con alguna de las keys."""
    try:
        for k in keys:
            if k in df.index:
                return df.loc[k]
    except Exception:
        pass
    # retornar Serie vacía compatible
    try:
        return pd.Series([0], index=df.columns[:1])
    except Exception:
        return pd.Series([0])

def format_number(x, decimals=2, is_percent=False):
    if pd.isna(x):
        return "N/D"
    try:
        if is_percent:
            return f"{x*100:.{decimals}f}%"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "N/D"

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    try:
        return interest / debt if debt else 0
    except Exception:
        return 0

def calc_wacc(mcap, debt, ke, kd, t):
    try:
        total = (mcap or 0) + (debt or 0)
        return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None
    except Exception:
        return None

def cagr4(fin, metric):
    """Calcula CAGR aproximado con hasta 4 periodos (si existen)."""
    try:
        if fin is None or fin.empty:
            return None
        if metric not in fin.index:
            return None
        v = fin.loc[metric].dropna().iloc[:4]
        if len(v) <= 1 or v.iloc[-1] == 0:
            return None
        return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1)) - 1
    except Exception:
        return None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y dado un DataFrame/array/Series."""
    try:
        if isinstance(values, pd.DataFrame) or isinstance(values, pd.Series):
            arr = values.to_numpy(dtype="float64")
        else:
            arr = np.asarray(values, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        if vmax == vmin:
            ymin = vmin - abs(vmin)*pad - 1
            ymax = vmax + abs(vmax)*pad + 1
            ax.set_ylim(ymin, ymax)
            return
        if vmin >= 0:
            ymin = 0
            ymax = vmax * (1 + pad)
        elif vmax <= 0:
            ymax = 0
            ymin = vmin * (1 + pad)
        else:
            m = max(abs(vmin), abs(vmax)) * (1 + pad)
            ymin, ymax = -m, m
        ax.set_ylim(ymin, ymax)
    except Exception:
        pass

def obtener_balance_historico(ticker, años=4):
    """Obtener datos históricos de balance para los últimos años (activos, pasivos, patrimonio)."""
    try:
        tkr = yf.Ticker(ticker)
        balance_sheets = tkr.balance_sheet
        if balance_sheets is None or balance_sheets.empty:
            return None

        fechas_disponibles = balance_sheets.columns[:min(años, len(balance_sheets.columns))]
        datos = {}
        for fecha in fechas_disponibles:
            try:
                año = fecha.year
            except Exception:
                año = str(fecha)
            activos = safe_first(seek_row(balance_sheets[fecha], [
                "Total Assets", "TotalAssets", "Total Current Assets"
            ])) or 0
            pasivos = safe_first(seek_row(balance_sheets[fecha], [
                "Total Liabilities", "Total Liabilities Net Minority Interest",
                "Total Current Liabilities"
            ])) or 0
            patrimonio = safe_first(seek_row(balance_sheets[fecha], [
                "Total Stockholder Equity", "Stockholders Equity",
                "Common Stock Equity"
            ])) or 0

            datos[año] = {
                "Activos Totales": activos,
                "Pasivos Totales": pasivos,
                "Patrimonio Neto": patrimonio
            }
        return datos
    except Exception as e:
        # No usar st.error dentro de la función si se llama muchas veces; devolver None
        return None

def obtener_datos_financieros(tk, Tc_def):
    """Obtiene los principales datos/rátios desde yfinance y calcula WACC/ROIC/etc."""
    try:
        tkr = yf.Ticker(tk)
        info = tkr.info or {}
        bs = tkr.balance_sheet if tkr.balance_sheet is not None else pd.DataFrame()
        fin = tkr.financials if tkr.financials is not None else pd.DataFrame()
        cf = tkr.cashflow if tkr.cashflow is not None else pd.DataFrame()

        beta = info.get("beta", 1)
        ke = calc_ke(beta)

        debt = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
        cash = safe_first(seek_row(bs, [
            "Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments",
        ])) or 0
        equity = safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"])) or 0

        interest = safe_first(seek_row(fin, ["Interest Expense"])) or 0
        ebt = safe_first(seek_row(fin, ["Ebt", "EBT"])) or 0
        tax_exp = safe_first(seek_row(fin, ["Income Tax Expense"])) or 0
        ebit = safe_first(seek_row(fin, ["EBIT", "Operating Income", "Earnings Before Interest and Taxes"]))

        kd = calc_kd(interest, debt)
        tax = (tax_exp / ebt) if ebt else Tc_def
        mcap = info.get("marketCap", 0)
        wacc = calc_wacc(mcap, debt, ke, kd, tax)

        nopat = (ebit * (1 - tax)) if (ebit is not None) else None
        invested = (equity or 0) + ((debt or 0) - (cash or 0))
        roic = nopat / invested if (nopat is not None and invested) else None
        creacion_valor = (roic - wacc) * 100 if all(v is not None for v in (roic, wacc)) else None

        price = info.get("currentPrice")
        fcf = safe_first(seek_row(cf, ["Free Cash Flow"])) or None
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf/shares) if (fcf and shares) else None

        # Ratios desde info (si existen)
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        debt_eq = info.get("debtToEquity")
        lt_debt_eq = info.get("longTermDebtToEquity")
        oper_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")

        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")

        revenue_growth = cagr4(fin, "Total Revenue")
        eps_growth = cagr4(fin, "Net Income")
        fcf_growth = cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow")

        return {
            "Ticker": tk,
            "Nombre": info.get("longName") or info.get("shortName") or info.get("displayName") or tk,
            "País": info.get("country") or info.get("countryCode") or "N/D",
            "Industria": info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D",
            "Sector": info.get("sector", "Unknown"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Yield %": div_yield,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Debt/Eq": debt_eq,
            "LtDebt/Eq": lt_debt_eq,
            "Oper Margin": oper_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "Creacion Valor (Wacc vs Roic)": creacion_valor,
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "MarketCap": mcap
        }
    except Exception as e:
        st.error(f"Error obteniendo datos para {tk}: {str(e)}")
        return None

# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (separados por comas)", 
                          "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, CB, DDOG")
        max_t = st.slider("Máximo de tickers", 1, 100, 50)
        
        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc0
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0)/100

    if st.button("🔍 Analizar Acciones", type="primary"):
        tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]
        
        # Obtener datos
        datos = []
        errs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Obteniendo datos financieros..."):
            for i, tk in enumerate(tickers):
                try:
                    status_text.text(f"⏳ Procesando {tk} ({i+1}/{len(tickers)})...")
                    data = obtener_datos_financieros(tk, Tc0)
                    if data:
                        datos.append(data)
                    else:
                        errs.append({"Ticker": tk, "Error": "Datos incompletos o no disponibles"})
                except Exception as e:
                    errs.append({"Ticker": tk, "Error": str(e)})
                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(1)  # evitar rate limiting básico

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not datos:
            st.error("No se pudieron obtener datos para los tickers proporcionados")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(datos)
        df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(99).astype(int)
        df = df.sort_values(["SectorRank", "Sector", "Ticker"])

        # Formateo para visualización
        df_disp = df.copy()
        for col in ["P/E", "P/B", "P/FCF", "Current Ratio", "Quick Ratio", "Debt/Eq", "LtDebt/Eq"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2))
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", 
                   "Profit Margin", "WACC", "ROIC", "Revenue Growth", "EPS Growth", "FCF Growth"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2, is_percent=True))
        df_disp["Creacion Valor (Wacc vs Roic)"] = df_disp["Creacion Valor (Wacc vs Roic)"].apply(
            lambda x: format_number(x/100, 2, is_percent=True) if pd.notnull(x) else "N/D"
        )
        df_disp["Precio"] = df_disp["Precio"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "N/D")
        df_disp["MarketCap"] = df_disp["MarketCap"].apply(lambda x: f"${float(x)/1e9:,.2f}B" if pd.notnull(x) else "N/D")
        for c in ["Nombre", "País", "Industria"]:
            df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # SECCIÓN 1: RESUMEN GENERAL
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        st.dataframe(
            df_disp[[
                "Ticker", "Nombre", "País", "Industria", "Sector",
                "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"
            ]],
            use_container_width=True,
            height=500
        )
        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df.sort_values("SectorRank")["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # =====================================================
        # SECCIÓN 3: RENTABILIDAD Y EFICIENCIA
        # =====================================================
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"].fillna(0)*100).values,
                        "ROA": (sec_df["ROA"].fillna(0)*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"].fillna(0)*100).values,
                        "Profit Margin": (sec_df["Profit Margin"].fillna(0)*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 6))
            rw = pd.DataFrame({
                "ROIC": (df["ROIC"].fillna(0)*100).values,
                "WACC": (df["WACC"].fillna(0)*100).values
            }, index=df["Ticker"])
            rw.plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            auto_ylim(ax, rw)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # =============================================================
        # SECCIÓN 4: ESTRUCTURA DE CAPITAL Y LIQUIDEZ
        # =============================================================
        st.header("🏦 Estructura de Capital y Liquidez (por sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.caption("Patrimonio Deuda Activos (Últimos 4 años)")
                        datos_historicos = {}
                        for _, empresa in chunk.iterrows():
                            ticker = empresa["Ticker"]
                            balance_data = obtener_balance_historico(ticker)
                            if balance_data:
                                datos_historicos[ticker] = balance_data
                        if datos_historicos:
                            # tamaño fijo igual al de Liquidez
                            fig, axes = plt.subplots(len(datos_historicos), 1, 
                                                    figsize=(10, 5),
                                                    sharex=True, sharey=True)
                            fig.suptitle(f"Estructura Patrimonial - Sector {sec}", fontsize=16)
                            if len(datos_historicos) == 1:
                                axes = [axes]
                            for idx, (ticker, datos) in enumerate(datos_historicos.items()):
                                ax = axes[idx]
                                años = sorted(datos.keys())
                                activos = [datos[a]["Activos Totales"]/1e6 for a in años]
                                pasivos = [datos[a]["Pasivos Totales"]/1e6 for a in años]
                                patrimonio = [datos[a]["Patrimonio Neto"]/1e6 for a in años]
                                x_pos = np.arange(len(años))
                                width = 0.25
                                ax.bar(x_pos - width, activos, width, label='Activos Totales', alpha=0.8, color='#87CEEB')
                                ax.bar(x_pos, pasivos, width, label='Pasivos Totales', alpha=0.8, color='#FFA07A')
                                ax.bar(x_pos + width, patrimonio, width, label='Patrimonio Neto', alpha=0.8, color='#32CD32')
                                ax.set_ylabel('Millones USD')
                                ax.set_title(f'{ticker}')
                                ax.set_xticks(x_pos)
                                ax.set_xticklabels(años)
                                ax.legend()
                                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                            plt.tight_layout(rect=[0, 0, 1, 0.92])
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info(f"No se pudieron obtener datos históricos para las empresas del sector {sec}")
                    
                    with c2:
                        st.caption("Liquidez")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        liq = chunk[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        liq.plot(kind="bar", ax=ax, rot=45)
                        ax.axhline(1, color="green", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, liq)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO (CAGR 3-4 años, por sector y por bloque)
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por sector)")
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    # Preparar DataFrame de crecimiento (multiplicar por 100 para %)
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"].fillna(0)*100).values,
                        "EPS Growth": (chunk["EPS Growth"].fillna(0)*100).values,
                        "FCF Growth": (chunk["FCF Growth"].fillna(0)*100).values
                    }, index=chunk["Ticker"])
                    fig, ax = plt.subplots(figsize=(12, 6))
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    ax.set_title(f"Crecimiento - Sector {sec} (Bloque {i})")
                    auto_ylim(ax, gdf)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

        # =====================================================
        # SECCIÓN 6: ANÁLISIS INDIVIDUAL POR EMPRESA
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        **{det_raw['Nombre']}**  
        **Sector:** {det_raw['Sector']}  
        **País:** {det_raw['País']}  
        **Industria:** {det_raw['Industria']}
        """)

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            
        with cB:
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            st.metric("Creación Valor", det_disp["Creacion Valor (Wacc vs Roic)"])
            
        with cC:
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(5, 4))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            # color condicional simplificado
            colors = []
            if det_raw["ROIC"] is not None and det_raw["WACC"] is not None and det_raw["ROIC"] > det_raw["WACC"]:
                colors = ["#2ca02c", "gray"]
            else:
                colors = ["#d62728", "gray"]
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, color=colors)
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

if __name__ == "__main__":
    main()
