import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import openai
import time
import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import json

# 🔹 Activar modo wide
st.set_page_config(layout="wide")
# Inyectar CSS para el estilo del chat (burbujas modernas)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-top: 20px;
        font-family: 'Roboto', sans-serif;
    }

    .message-row {
        display: flex;
        width: 100%;
    }

    .user-message {
        background: #081c15;
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 0;
        max-width: 60%;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-right: auto;
        margin-bottom: 20px;
    }

    .bot-message {
        background: #344e41;
        color: #ddd;
        padding: 15px 20px;
        border-radius: 20px 20px 0 20px;
        max-width: 60%;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-left: auto;
        margin-bottom: 20px;
    }

    .user-message:hover, .bot-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .user-message, .bot-message {
        animation: fadeIn 0.3s ease-out;
    }
    </style>
    """, unsafe_allow_html=True
)

# 🔹 Agregar título
#st.title("💰Panel de control de Ingresos - Internet y Combazos")

# 🔹 Configurar las credenciales desde Streamlit Secrets
credentials_data = st.secrets["google"]["credentials"]
credentials_dict = json.loads(credentials_data)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# 🔹 Autenticación con Google Sheets
creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
client = gspread.authorize(creds)

# 🔹 Abrir la hoja de cálculo
spreadsheet_url = "https://docs.google.com/spreadsheets/d/1144jZxmPrd9Vt_zTB5ZY4KljHEZ3eFlmJlpPP5uN85g"
spreadsheet = client.open_by_url(spreadsheet_url)



# 🔹 Función para obtener datos de una hoja específica
def get_sheet_data(sheet_index):
    sheet = spreadsheet.get_worksheet(sheet_index)  # Obtener la hoja por índice
    data = sheet.get_all_records()  # Obtener datos
    return pd.DataFrame(data)  # Convertir a DataFrame

# Configurar el modelo de OpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0.3, 
    openai_api_key = st.secrets["openai"]["api_key"]
)

def get_calculation_result(query, df):
    prompt_code = f"""
Tienes un DataFrame llamado 'df' con las siguientes columnas: {', '.join(df.columns)}.
El usuario ha preguntado: "{query}".
Genera únicamente el código en Python (sin explicaciones ni comentarios) que realice el cálculo solicitado y almacene el resultado en una variable llamada result.
    """
    code_response = llm.predict(prompt_code).strip()
    
    local_vars = {}
    try:
        exec(code_response, {"df": df}, local_vars)
        result = local_vars.get("result", None)
    except Exception as e:
        result = None
    return result

def get_natural_language_answer(query, result):
    prompt_natural = f"""
El usuario ha preguntado: "{query}".
El resultado del cálculo es: {result}.
Responde de forma clara y en lenguaje natural, sin mostrar el código, indicando el resultado de la consulta.
    """
    natural_response = llm.predict(prompt_natural).strip()
    return natural_response

def summarize_dataframe(df):
    """Genera un resumen sencillo del DataFrame para dar contexto."""
    if df.empty:
        return "⚠️ No hay datos disponibles."
    sample_data = df.head(5).to_dict(orient="records")
    aggregates = {
        "Total de registros": len(df),
        "Columnas": list(df.columns)
    }
    summary = f"Métricas clave:\n{aggregates}\n\nEjemplo de datos:\n{sample_data}"
    return summary

def get_analysis_opinion(df):
    """Genera una opinión y análisis global del DataFrame a partir de un resumen."""
    summary = summarize_dataframe(df)
    prompt_analysis = f"""
Eres un analista de datos experto. A partir del siguiente resumen del DataFrame:
{summary}
Da tu opinión y análisis sobre las tendencias, patrones y aspectos destacados del archivo, en un lenguaje claro y conciso.
    """
    opinion = llm.predict(prompt_analysis).strip()
    return opinion

def get_analysis_by_column(df):
    """Genera un resumen y análisis para cada columna del DataFrame."""
    analysis_result = {}
    for col in df.columns:
        prompt = f"""
Eres un experto en análisis de datos. Analiza la columna '{col}' del DataFrame.
Proporciona un resumen claro y conciso de su contenido, tendencias y cualquier información relevante.
Si la columna es numérica, incluye estadísticas básicas; si es categórica, menciona los valores más frecuentes.
        """
        analysis_result[col] = llm.predict(prompt).strip()
    return analysis_result

def is_relevant_query(query, df):
    """Verifica si la consulta está relacionada con las columnas del DataFrame o contiene palabras clave de análisis."""
    query_lower = query.lower()
    columns_lower = [col.lower() for col in df.columns]
    if any(col in query_lower for col in columns_lower):
        return True
    keywords = ["suma", "promedio", "total", "cuántos", "filtrar", "estadísticas", "análisis", "dato", "mínimo", "máximo"]
    if any(keyword in query_lower for keyword in keywords):
        return True
    return False

def is_analysis_by_column(query):
    """Determina si se solicita un análisis por cada columna."""
    keywords = ["por cada columna", "analiza cada columna", "resumen por columna", "análisis por columna"]
    return any(keyword in query.lower() for keyword in keywords)

def is_greeting(query):
    """Determina si la consulta es un saludo."""
    greetings = ["hola", "buenos días", "buenas tardes", "saludos"]
    return any(greet in query.lower() for greet in greetings)

def is_farewell(query):
    """Determina si la consulta es una despedida."""
    farewells = ["adiós", "hasta luego", "nos vemos", "chau", "chao"]
    return any(farewell in query.lower() for farewell in farewells)

# st.title("📊 Chatbot de Análisis de Datos con Pandas - INICIO")
st.sidebar.title("Menú de Navegación")
option = st.sidebar.selectbox("Selecciona una opción:", ["INICIO", "CHAT IA", "POSTPAGO", "PREPAGO"])

if option == "INICIO":
    st.title("📊 Bienvenido al Panel de Control")
    st.write("Selecciona una opción en el menú lateral para ver los datos.")

if option == "CHAT IA":
    st.title("🤖 DataIA - Análisis de Datos (Test)")
    selected_sheet = st.selectbox("📂 Selecciona el tipo de datos", ["Postpago", "Prepago"])
    sheet_index = 0 if selected_sheet == "Postpago" else 1

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    df = get_sheet_data(sheet_index)
    if not df.empty:
        column_names = df.columns.tolist()
        st.subheader("Columnas Disponibles:")
        st.write(column_names)
    
    chat_container = st.empty()
    with chat_container.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["sender"] == "user":
                st.markdown(f'<div class="user-message"><strong>Tú:</strong> {message["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {message["message"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 1, 1])

    with col1:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("💬 Escribe tu pregunta y presiona Enter")
            submit_button = st.form_submit_button(label="Enviar")

    with col2:
        if st.button("Limpiar Chat"):
            st.session_state.chat_history = []
            chat_container.empty()

    with col3:
        if st.button("Descargar Chat"):
            chat_text = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in st.session_state.chat_history])
            st.download_button(
                label="Descargar chat como .txt",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    if submit_button and user_input:
        st.session_state.chat_history.append({"sender": "user", "message": user_input})
        
        with st.spinner("⏳ Procesando tu consulta..."):
            # Si es una despedida, responde y finaliza
            if is_farewell(user_input):
                answer = "¡Adiós! Gracias por usar DataIA. ¡Hasta pronto! 👋"
            # Si es un saludo pero no tiene otros elementos relevantes, responde con saludo
            elif is_greeting(user_input) and not is_relevant_query(user_input, df):
                answer = "¡Hola! ¿En qué puedo ayudarte con el análisis de datos?"
            # Si se solicita un análisis global del archivo
            elif "analiza el archivo" in user_input.lower() or "análisis del archivo" in user_input.lower():
                answer = get_analysis_opinion(df)
                if is_greeting(user_input):
                    answer = "¡Hola! " + answer
            # Si se solicita un análisis por cada columna
            elif is_analysis_by_column(user_input):
                analysis = get_analysis_by_column(df)
                analysis_str = ""
                for col, text in analysis.items():
                    analysis_str += f"**{col}:**\n{text}\n\n"
                answer = analysis_str
                if is_greeting(user_input):
                    answer = "¡Hola! " + answer
            # Si la consulta es relevante para cálculos
            elif is_relevant_query(user_input, df):
                result = get_calculation_result(user_input, df)
                if result is None:
                    answer = "Soy un bot en entrenamiento, pero tomaré en cuenta tu consulta para mejorar mi eficiencia."
                else:
                    answer = get_natural_language_answer(user_input, result)
                if is_greeting(user_input):
                    answer = "¡Hola! " + answer
            else:
                answer = "Lo siento, solo puedo responder preguntas relacionadas con el análisis de datos del archivo. 😊"
        
        st.session_state.chat_history.append({"sender": "bot", "message": answer})
        
        # Actualizar el contenedor del chat
        chat_container.empty()
        with chat_container.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.chat_history:
                if message["sender"] == "user":
                    st.markdown(f'<div class="user-message"><strong>Tú:</strong> {message["message"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message"><strong>Bot:</strong> {message["message"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

elif option == "POSTPAGO":
    st.title("💰 Panel de Control - POSTPAGO")
    df = get_sheet_data(0)  # Primera hoja (Postpago)

    # 🔹 Convertir "FECHA DE REVISION" a formato de fecha
    df["FECHA DE REVISION"] = pd.to_datetime(df["FECHA DE REVISION"], errors="coerce")
    
    # 🔹 Obtener la fecha más reciente
    fecha_maxima = df["FECHA DE REVISION"].max()
    
    # 🔹 Mostrar la fecha máxima en st.info() debajo del título
    if pd.notna(fecha_maxima):
        st.info(f"📅 Reporte actualizado al: **{fecha_maxima.strftime('%d-%m-%Y')}**")
    else:
        st.warning("⚠️ No se encontraron fechas válidas en la columna 'FECHA DE REVISION'.")
    
    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Mostrar/Ocultar Filtros", expanded=False):  # expanded=False para que esté colapsado por defecto
        st.header("Filtros")
    
        # 1. Filtro por CONTRATO
        if 'CONTRATO' in df.columns:
            contratos = df['CONTRATO'].unique().tolist()
            selected_contratos = st.multiselect(
                'CONTRATO',
                options=contratos,
                default=None,
                help="Selecciona uno o varios contratos"
            )
            if selected_contratos:
                df = df[df['CONTRATO'].isin(selected_contratos)]
    
        # 2. Filtro por FECHA DE ENVIO DE INFORME (versión mejorada)
        if 'FECHA ENVIO' in df.columns:
            try:
                # Convertir a formato fecha y extraer únicas
                df['FECHA_TS'] = pd.to_datetime(df['FECHA ENVIO'])
                fechas_unicas = df['FECHA_TS'].dt.date.unique().tolist()
                fechas_ordenadas = sorted(fechas_unicas, reverse=True)
                
                selected_fechas = st.multiselect(
                    'FECHA ENVIO',
                    options=['Seleccionar todas'] + fechas_ordenadas,
                    default=['Seleccionar todas'],
                    format_func=lambda x: x.strftime('%d/%m/%Y') if isinstance(x, datetime.date) else x
                )
                
                if 'Seleccionar todas' not in selected_fechas:
                    df = df[df['FECHA_TS'].dt.date.isin(selected_fechas)]
            except Exception as e:
                st.error(f"Formato de fecha no reconocido: {str(e)}")
                st.write("Valores únicos en la columna:", df['FECHA ENVIO'].unique())
    
        # 3. Filtro por TIPO DE SERVICIO
        if 'TIPO DE SERVICIO' in df.columns:
            servicios = df['TIPO DE SERVICIO'].unique().tolist()
            selected_servicios = st.multiselect(
                'TIPO DE SERVICIO',
                options=servicios,
                default=None,
                help="Filtra por tipo de servicio"
            )
            if selected_servicios:
                df = df[df['TIPO DE SERVICIO'].isin(selected_servicios)]
    
        # 4. Filtro por ESTADO
        if 'ESTADO' in df.columns:
            estados = df['ESTADO'].unique().tolist()
            selected_estados = st.multiselect(
                'ESTADO',
                options=estados,
                default=None,
                help="Filtra por estado del caso"
            )
            if selected_estados:
                df = df[df['ESTADO'].isin(selected_estados)]
    
        # 5. Filtro por RESPONSABLE
        if 'RESPONSABLE' in df.columns:
            responsables = df['RESPONSABLE'].unique().tolist()
            selected_responsables = st.multiselect(
                'RESPONSABLE',
                options=responsables,
                default=None,
                help="Filtra por responsable asignado"
            )
            if selected_responsables:
                df = df[df['RESPONSABLE'].isin(selected_responsables)]
    
        # 6. Filtro por TIPO DE ERROR
        if 'TIPO DE ERROR' in df.columns:
            errores = df['TIPO DE ERROR'].unique().tolist()
            selected_errores = st.multiselect(
                'TIPO DE ERROR',
                options=errores,
                default=None,
                help="Filtra por tipo de error reportado"
            )
            if selected_errores:
                df = df[df['TIPO DE ERROR'].isin(selected_errores)]
    
    # 🔹 Cálculo de métricas principales
    total_contratos = df["CONTRATO"].count()
    monto_recuperado = df["MONTO RECUPERADO"].sum()
    monto_no_recuperado = df["MONTO NO RECUPERADO"].sum()
    promedio_periodos_no_facturados = df["PERIODOS NO FACTURADO"].mean()
    
    # 🔹 Lista de tarjetas con métricas personalizadas
    tarjetas = [
        {"titulo": "Contratos Reportados", "valor": total_contratos, "color": "#3b5465", "hover": "#99311a"},
        {"titulo": "Monto Recuperado", "valor": f"Bs {monto_recuperado:,.2f}", "color": "#274029", "hover": "#1f4a04"},
        {"titulo": "Monto No Recuperado", "valor": f"Bs {monto_no_recuperado:,.2f}", "color": "#11181d", "hover": "#3b0610"},
    ]
    
    # 🔹 Generar el CSS dinámico
    css_tarjetas = ""
    for i, tarjeta in enumerate(tarjetas):
        css_tarjetas += f"""
            .card-{i} {{
                background-color: {tarjeta['color']};
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 150px;
                width: 100%;
                padding: 20px;
                border-radius: 20px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
            }}
            .card-{i}:hover {{
                background-color: {tarjeta['hover']};
                transform: scale(1.05);
                box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.3);
            }}
        """
    
    # 🔹 Aplicar CSS en Streamlit
    st.markdown(f"""
        <style>
            .custom-card {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                width: 100%;
                height: 150px;
                padding: 20px;
                border-radius: 20px;
                border: 2px solid #ffffff;
            }}
            {css_tarjetas}
    
            /* Asegurarse de que todos los valores estén alineados de la misma forma */
            .custom-card p {{
                font-size: 34px;
                font-weight: bold;
                color: white;
                margin: 0;
                text-align: center;
            }}
            
            /* Estilo para centrar las nuevas métricas */
            .centered-metrics {{
                display: flex;
                justify-content: center;
                margin-top: 40px;
            }}
            .metric-box {{
                padding: 20px;
                margin: 10px;
                border-radius: 10px;
                background-color: #f0f2f6;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                min-width: 250px;
                text-align: center;
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # 🔹 Mostrar tarjetas principales en columnas
    num_columnas = len(tarjetas)
    columnas = st.columns(num_columnas)
    
    for i, tarjeta in enumerate(tarjetas):
        with columnas[i]:  
            metric_placeholder = st.empty()
            
            # Animación solo para Contratos Reportados
            if tarjeta["titulo"] == "Contratos Reportados":
                valor_final = tarjeta["valor"]
                for j in range(valor_final + 1):
                    metric_placeholder.markdown(
                        f"""
                        <div class="custom-card card-{i}">
                            <h3 style="color: white; margin-bottom: 10px; text-align: center;">{tarjeta['titulo']}</h3>
                            <p>{j}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)
            else:
                metric_placeholder.markdown(
                    f"""
                    <div class="custom-card card-{i}">
                        <h3 style="color: white; margin-bottom: 10px; text-align: center;">{tarjeta['titulo']}</h3>
                        <p>{tarjeta['valor']}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    
    # 🔹 Nueva sección de métricas centradas
    st.markdown("""
    <style>
        .metric-spacer {
            margin-top: 0px !important;
        }
        [data-testid="stMetric"] {
            width: 80% !important;
            padding: 0 15px !important;
        }
        [data-testid="stMetricLabel"] {
            width: 100% !important;
            
            font-size: 14px !important;
            text-align: center !important;
            margin-bottom: 8px !important;
            display: block !important;
            padding: 0 !important;
        }
        [data-testid="stMetricValue"] {
            width: 100% !important;
            color: #FF5733 !important;
            font-size: 42px !important;
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        [data-testid="column"] {
            align-items: center !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Calcular nuevas métricas
    try:
        # Casos Resueltos (cuenta de ESTADO = RESUELTO)
        casos_resueltos = df[df['ESTADO'] == 'RESUELTO'].shape[0]
    except KeyError:
        casos_resueltos = 0
    
    try:
        # Casos Pendientes (cuenta de ESTADO = PENDIENTE)
        casos_pendientes = df[df['ESTADO'] == 'PENDIENTE'].shape[0]
    except KeyError:
        casos_pendientes = 0
    
    try:
        promedio_dias_demora = df["DIAS DE DEMORA"].mean()
    except KeyError:
        promedio_dias_demora = 0
    
    # Espaciado superior y contenedor centrado
    st.markdown("<div class='metric-spacer'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([0.1, 3, 0.1])
    
    with center_col:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="CASOS RESUELTOS",
                value=casos_resueltos
            )
        
        with col2:
            st.metric(
                label="CASOS PENDIENTES",
                value=casos_pendientes
            )
        
        with col3:
            st.metric(
                label="PROMEDIO DE RESPUESTA DE OBSERVACIONES",
                value=f"{promedio_dias_demora:.2f} días"  # Formato con 2 decimales
            )
    
    
    
    
    st.markdown(
        '<p style="background-color:#001a2c; color:whithe; padding:10px; border-radius:5px;">'
        '📊 <b>Area de graficos</b>'
        '</p>',
        unsafe_allow_html=True
    )
    # 🔹 Sección NO FACTURADO MENSUAL OBSERVADO
    #st.header("📊 Observacion Mensual no facturado")
    
    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Monto no recuperado filtrado por fecha del envio de la observacion", expanded=False):  # expanded=False para que esté colapsado por defecto
        #st.header("Filtros")
    
        # Selector de tipo de gráfico
        tipo_grafico = st.radio("Selecciona el tipo de gráfico:", ["Línea", "Barras"], horizontal=True)
    
        if 'TARIFA PLAN' in df.columns and 'FECHA_TS' in df.columns:
            # Convertir a tipo numérico y limpiar datos
            df['TARIFA PLAN'] = pd.to_numeric(df['TARIFA PLAN'], errors='coerce')
            df['FECHA_TS'] = pd.to_datetime(df['FECHA_TS'], errors='coerce')
            df_valid = df.dropna(subset=['TARIFA PLAN', 'FECHA_TS'])
            
            if not df_valid.empty:
                # Agrupar y sumar por fecha
                df_sum = df_valid.groupby('FECHA_TS', as_index=False)['TARIFA PLAN'].sum()
                
                # Crear gráfico
                fig = go.Figure()
                
                if tipo_grafico == "Línea":
                    # Línea principal
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['TARIFA PLAN'], 
                        mode='lines+markers',
                        name='Suma Tarifa Plan',
                        line=dict(color='#04514f', width=6),
                        marker=dict(size=8, color='darkblue', line=dict(width=1, color='white'))
                    ))
    
                    # Línea de tendencia (Media móvil de 7 días)
                    df_sum['Tendencia'] = df_sum['TARIFA PLAN'].rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['Tendencia'], 
                        mode='lines',
                        name='Tendencia (Media 7 días)',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                
                else:
                    # Agregar un selector de color
                    color_barras = st.color_picker("Elige el color de las barras", "#04514f")
    
                    fig.add_trace(go.Bar(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['TARIFA PLAN'], 
                        name='Suma Tarifa Plan',
                        marker=dict(color=color_barras)  # Aplica el color elegido
                    ))
    
    
                # Configuración del diseño
                fig.update_layout(
                    title='📊 Monto no recuperado por Fecha de Envío',
                    xaxis_title='Fecha de Envío',
                    yaxis_title='Suma Acumulada (Bs)',
                    template='plotly_white',
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    legend=dict(title='Indicadores', x=0.01, y=1.15, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos válidos para mostrar el gráfico.")
        elif 'TARIFA PLAN' not in df.columns:
            st.warning("⚠️ La columna 'TARIFA PLAN' no existe en el dataset.")
        else:
            st.warning("⚠️ Problema con las fechas: verifique la columna 'FECHA_TS'.")
    
    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Monto recuperado filtrado por fecha del envio de la observacion ", expanded=False):  # expanded=False para que esté colapsado por defecto
        #st.header("Filtros")
    
        # Selector de tipo de gráfico
        tipo_grafico = st.radio("Selecciona el tipo de gráfico:", ["Línea", "Barras"], horizontal=True,key="boton1")
    
        if 'MONTO RECUPERADO' in df.columns and 'FECHA ENVIO' in df.columns:
            # Convertir a tipo numérico y limpiar datos
            df['MONTO RECUPERADO'] = pd.to_numeric(df['MONTO RECUPERADO'], errors='coerce')
            df['FECHA ENVIO'] = pd.to_datetime(df['FECHA ENVIO'], errors='coerce')
            df_valid = df.dropna(subset=['MONTO RECUPERADO', 'FECHA ENVIO'])
            
            if not df_valid.empty:
                # Agrupar y sumar por fecha
                df_sum = df_valid.groupby('FECHA ENVIO', as_index=False)['MONTO RECUPERADO'].sum()
                
                # Crear gráfico
                fig = go.Figure()
                
                if tipo_grafico == "Línea":
                    # Línea principal
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA ENVIO'], 
                        y=df_sum['MONTO RECUPERADO'], 
                        mode='lines+markers',
                        name='Suma Monto Recuperado',
                        line=dict(color='#04514f', width=6),
                        marker=dict(size=8, color='darkblue', line=dict(width=1, color='white'))
                    ))
    
                    # Línea de tendencia (Media móvil de 7 días)
                    df_sum['Tendencia'] = df_sum['MONTO RECUPERADO'].rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA ENVIO'], 
                        y=df_sum['Tendencia'], 
                        mode='lines',
                        name='Tendencia (Media 7 días)',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                
                else:
                    # Agregar un selector de color
                    color_barras = st.color_picker("Elige el color de las barras", "#04514f",key="color1")
    
                    fig.add_trace(go.Bar(
                        x=df_sum['FECHA ENVIO'], 
                        y=df_sum['MONTO RECUPERADO'], 
                        name='Suma Monto Recuperado',
                        marker=dict(color=color_barras)  # Aplica el color elegido
                    ))
    
    
                # Configuración del diseño
                fig.update_layout(
                    title='📊 Evolución de Monto Recuperado por Fecha de Envío',
                    xaxis_title='Fecha de Envío',
                    yaxis_title='Monto Recuperado (Bs)',
                    template='plotly_white',
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    legend=dict(title='Indicadores', x=0.01, y=1.15, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos válidos para mostrar el gráfico.")
        elif 'MONTO RECUPERADO' not in df.columns:
            st.warning("⚠️ La columna 'MONTO RECUPERADO' no existe en el dataset.")
        else:
            st.warning("⚠️ Problema con las fechas: verifique la columna 'FECHA ENVIO'.")
    
    st.markdown(
        '<p style="background-color:#001a2c; color:whithe; padding:10px; border-radius:5px;">'
        '📊 <b>Area de reporte</b>'
        '</p>',
        unsafe_allow_html=True
    )
    with st.expander("🧩 Descargar Reporte", expanded=False):  # expanded=False para que esté colapsado por defecto
    # 🔹 Título atractivo para la sección
        #st.title("🔄 Automatizar Reporte")
    
        # Agregar un toque visual con un subtítulo y estilo
        st.markdown("""
            <style>
            .reportview-container {
                background-color: #f4f6f9;
                font-family: 'Arial', sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #2E4053;
            }
            .css-1d391kg {
                color: #2D3436;
            }
            </style>
            """, unsafe_allow_html=True)
        st.subheader("Filtros de datos y opciones de descarga")
    
        # Mostrar el DataFrame filtrado con st.dataframe
        st.dataframe(df)
    
        # 🔹 Opción para descargar el archivo filtrado
    # 🔹 Opción para descargar el archivo filtrado automáticamente
    @st.cache_data
    def to_excel(df):
        """Convierte el DataFrame a un archivo Excel en formato bytes."""
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Datos Filtrados")
            writer._save()  # Guardar el archivo Excel
        return towrite.getvalue()
    
    # Convertir el DataFrame filtrado a Excel
    excel_file = to_excel(df)
    
    # Descargar automáticamente el archivo Excel
    st.download_button(
        label="📥 Descargar archivo Excel",
        data=excel_file,
        file_name="reporte_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
elif option == "PREPAGO":
    st.title("💰 Panel de Control - PREPAGO")
    df = get_sheet_data(1)  # Segunda hoja (Prepago)

  

    
        # 🔹 Convertir "FECHA DE REVISION" a formato de fecha
    df["FECHA DE REVISION"] = pd.to_datetime(df["FECHA DE REVISION"], errors="coerce")

    # 🔹 Obtener la fecha más reciente
    fecha_maxima = df["FECHA DE REVISION"].max()

    # 🔹 Mostrar la fecha máxima en st.info() debajo del título
    if pd.notna(fecha_maxima):
        st.info(f"📅 Reporte actualizado al: **{fecha_maxima.strftime('%d-%m-%Y')}**")
    else:
        st.warning("⚠️ No se encontraron fechas válidas en la columna 'FECHA DE REVISION'.")


    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Mostrar/Ocultar Filtros", expanded=False):  # expanded=False para que esté colapsado por defecto
        st.header("Filtros")

        # 1. Filtro por CONTRATO
        if 'CONTRATO' in df.columns:
            contratos = df['CONTRATO'].unique().tolist()
            selected_contratos = st.multiselect(
                'CONTRATO',
                options=contratos,
                default=None,
                help="Selecciona uno o varios contratos"
            )
            if selected_contratos:
                df = df[df['CONTRATO'].isin(selected_contratos)]

        # 2. Filtro por FECHA DE ENVIO DE INFORME (versión mejorada)
        if 'FECHA ENVIO' in df.columns:
            try:
                # Convertir a formato fecha con el formato correcto
                df['FECHA_TS'] = pd.to_datetime(df['FECHA ENVIO'], format='%d/%m/%Y', errors='coerce')
                
                # Verificar si hay valores nulos (fechas no convertidas)
                if df['FECHA_TS'].isnull().any():
                    st.warning("⚠️ Algunas fechas no pudieron ser convertidas. Verifica el formato de 'FECHA ENVIO'.")
                    st.write("Valores problemáticos:", df[df['FECHA_TS'].isnull()]['FECHA ENVIO'].unique())
                
                # Extraer fechas únicas
                fechas_unicas = df['FECHA_TS'].dt.date.unique().tolist()
                fechas_ordenadas = sorted(fechas_unicas, reverse=True)
                
                # Selector de fechas
                selected_fechas = st.multiselect(
                    'FECHA ENVIO',
                    options=['Seleccionar todas'] + fechas_ordenadas,
                    default=['Seleccionar todas'],
                    format_func=lambda x: x.strftime('%d/%m/%Y') if isinstance(x, datetime.date) else x
                )
                
                # Filtrar por fechas seleccionadas
                if 'Seleccionar todas' not in selected_fechas:
                    df = df[df['FECHA_TS'].dt.date.isin(selected_fechas)]
            except Exception as e:
                st.error(f"Error al procesar las fechas: {str(e)}")
                st.write("Valores únicos en la columna 'FECHA ENVIO':", df['FECHA ENVIO'].unique())
        else:
            st.error("⚠️ La columna 'FECHA ENVIO' no existe en el DataFrame.")
            st.write("Columnas disponibles:", df.columns.tolist())

        # 3. Filtro por TIPO DE SERVICIO
        if 'TIPO DE SERVICIO' in df.columns:
            servicios = df['TIPO DE SERVICIO'].unique().tolist()
            selected_servicios = st.multiselect(
                'TIPO DE SERVICIO',
                options=servicios,
                default=None,
                help="Filtra por tipo de servicio"
            )
            if selected_servicios:
                df = df[df['TIPO DE SERVICIO'].isin(selected_servicios)]

        # 4. Filtro por ESTADO
        if 'ESTADO' in df.columns:
            estados = df['ESTADO'].unique().tolist()
            selected_estados = st.multiselect(
                'ESTADO',
                options=estados,
                default=None,
                help="Filtra por estado del caso"
            )
            if selected_estados:
                df = df[df['ESTADO'].isin(selected_estados)]

        # 5. Filtro por RESPONSABLE
        if 'RESPONSABLE' in df.columns:
            responsables = df['RESPONSABLE'].unique().tolist()
            selected_responsables = st.multiselect(
                'RESPONSABLE',
                options=responsables,
                default=None,
                help="Filtra por responsable asignado"
            )
            if selected_responsables:
                df = df[df['RESPONSABLE'].isin(selected_responsables)]

        # 6. Filtro por TIPO DE ERROR
        if 'TIPO DE ERROR' in df.columns:
            errores = df['TIPO DE ERROR'].unique().tolist()
            selected_errores = st.multiselect(
                'TIPO DE ERROR',
                options=errores,
                default=None,
                help="Filtra por tipo de error reportado"
            )
            if selected_errores:
                df = df[df['TIPO DE ERROR'].isin(selected_errores)]

    # 🔹 Cálculo de métricas principales
    total_contratos = df["CONTRATO"].count()
    monto_recuperado = df.loc[df["PAGO DESPUES DE LA OBSERVACION"] == "SI", "TARIFA PLAN"].sum()
    monto_no_recuperado = df["MONTO NO FACTURADO"].sum()
    promedio_periodos_no_facturados = df["PERIODOS NO FACTURADO"].mean()

    # 🔹 Lista de tarjetas con métricas personalizadas
    tarjetas = [
        {"titulo": "Contratos Reportados", "valor": total_contratos, "color": "#3b5465", "hover": "#99311a"},
        {"titulo": "Monto Recuperado", "valor": f"Bs {monto_recuperado:,.2f}", "color": "#274029", "hover": "#1f4a04"},
        {"titulo": "Monto No Recuperado", "valor": f"Bs {monto_no_recuperado:,.2f}", "color": "#11181d", "hover": "#4e0923"},
    ]

    # 🔹 Generar el CSS dinámico
    css_tarjetas = ""
    for i, tarjeta in enumerate(tarjetas):
        css_tarjetas += f"""
            .card-{i} {{
                background-color: {tarjeta['color']};
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 150px;
                width: 100%;
                padding: 20px;
                border-radius: 20px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
            }}
            .card-{i}:hover {{
                background-color: {tarjeta['hover']};
                transform: scale(1.05);
                box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.3);
            }}
        """

    # 🔹 Aplicar CSS en Streamlit
    st.markdown(f"""
        <style>
            .custom-card {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                width: 100%;
                height: 150px;
                padding: 20px;
                border-radius: 20px;
                border: 4px solid #003049;
            }}
            {css_tarjetas}

            /* Asegurarse de que todos los valores estén alineados de la misma forma */
            .custom-card p {{
                font-size: 34px;
                font-weight: bold;
                color: white;
                margin: 0;
                text-align: center;
            }}
            
            /* Estilo para centrar las nuevas métricas */
            .centered-metrics {{
                display: flex;
                justify-content: center;
                margin-top: 40px;
            }}
            .metric-box {{
                padding: 20px;
                margin: 10px;
                border-radius: 10px;
                background-color: #f0f2f6;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                min-width: 250px;
                text-align: center;
            }}
        </style>
    """, unsafe_allow_html=True)

    # 🔹 Mostrar tarjetas principales en columnas
    num_columnas = len(tarjetas)
    columnas = st.columns(num_columnas)

    for i, tarjeta in enumerate(tarjetas):
        with columnas[i]:  
            metric_placeholder = st.empty()
            
            # Animación solo para Contratos Reportados
            if tarjeta["titulo"] == "Contratos Reportados":
                valor_final = tarjeta["valor"]
                for j in range(valor_final + 1):
                    metric_placeholder.markdown(
                        f"""
                        <div class="custom-card card-{i}">
                            <h3 style="color: white; margin-bottom: 10px; text-align: center;">{tarjeta['titulo']}</h3>
                            <p>{j}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)
            else:
                metric_placeholder.markdown(
                    f"""
                    <div class="custom-card card-{i}">
                        <h3 style="color: white; margin-bottom: 10px; text-align: center;">{tarjeta['titulo']}</h3>
                        <p>{tarjeta['valor']}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # 🔹 Nueva sección de métricas centradas
    st.markdown("""
    <style>
        .metric-spacer {
            margin-top: 0px !important;
        }
        [data-testid="stMetric"] {
            width: 80% !important;
            padding: 0 15px !important;
        }
        [data-testid="stMetricLabel"] {
            width: 100% !important;
            
            font-size: 14px !important;
            text-align: center !important;
            margin-bottom: 8px !important;
            display: block !important;
            padding: 0 !important;
        }
        [data-testid="stMetricValue"] {
            width: 100% !important;
            color: #FF5733 !important;
            font-size: 42px !important;
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        [data-testid="column"] {
            align-items: center !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Calcular nuevas métricas
    try:
        # Casos Resueltos (cuenta de ESTADO = RESUELTO)
        casos_resueltos = df[df['ESTADO'] == 'RESUELTO'].shape[0]
    except KeyError:
        casos_resueltos = 0

    try:
        # Casos Pendientes (cuenta de ESTADO = PENDIENTE)
        casos_pendientes = df[df['ESTADO'] == 'PENDIENTE'].shape[0]
    except KeyError:
        casos_pendientes = 0

    try:
        promedio_dias_demora = df["DIAS DE DEMORA"].mean()
    except KeyError:
        promedio_dias_demora = 0

    # Espaciado superior y contenedor centrado
    st.markdown("<div class='metric-spacer'></div>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([0.1, 3, 0.1])

    with center_col:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="CASOS RESUELTOS",
                value=casos_resueltos
            )
        
        with col2:
            st.metric(
                label="CASOS PENDIENTES",
                value=casos_pendientes
            )
        
        with col3:
            st.metric(
                label="PROMEDIO DE RESPUESTA DE OBSERVACIONES",
                value=f"{promedio_dias_demora:.2f} días"  # Formato con 2 decimales
            )



    st.markdown(
        '<p style="background-color:#00111c; color:whithe; padding:10px; border-radius:5px;">'
        '📊 <b>Area de graficos</b>'
        '</p>',
        unsafe_allow_html=True
    )

    # 🔹 Sección NO FACTURADO MENSUAL OBSERVADO
    #st.header("📊 Observacion Mensual no facturado")

    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Monto no recuperado filtrado por fecha del envio de la observacion", expanded=False):  # expanded=False para que esté colapsado por defecto
        #st.header("Filtros")

        # Selector de tipo de gráfico
        tipo_grafico = st.radio("Selecciona el tipo de gráfico:", ["Línea", "Barras"], horizontal=True)

        if 'TARIFA PLAN' in df.columns and 'FECHA_TS' in df.columns:
            # Convertir a tipo numérico y limpiar datos
            df['TARIFA PLAN'] = pd.to_numeric(df['TARIFA PLAN'], errors='coerce')
            df['FECHA_TS'] = pd.to_datetime(df['FECHA_TS'], errors='coerce')
            df_valid = df.dropna(subset=['TARIFA PLAN', 'FECHA_TS'])
            
            if not df_valid.empty:
                # Agrupar y sumar por fecha
                df_sum = df_valid.groupby('FECHA_TS', as_index=False)['TARIFA PLAN'].sum()
                
                # Crear gráfico
                fig = go.Figure()
                
                if tipo_grafico == "Línea":
                    # Línea principal
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['TARIFA PLAN'], 
                        mode='lines+markers',
                        name='Suma Tarifa Plan',
                        line=dict(color='#04514f', width=6),
                        marker=dict(size=8, color='darkblue', line=dict(width=1, color='white'))
                    ))

                    # Línea de tendencia (Media móvil de 7 días)
                    df_sum['Tendencia'] = df_sum['TARIFA PLAN'].rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['Tendencia'], 
                        mode='lines',
                        name='Tendencia (Media 7 días)',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                
                else:
                    # Agregar un selector de color
                    color_barras = st.color_picker("Elige el color de las barras", "#04514f")

                    fig.add_trace(go.Bar(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['TARIFA PLAN'], 
                        name='Suma Tarifa Plan',
                        marker=dict(color=color_barras)  # Aplica el color elegido
                    ))


                # Configuración del diseño
                fig.update_layout(
                    title='📊 Monto no recuperado por Fecha de Envío',
                    xaxis_title='Fecha de Envío',
                    yaxis_title='Suma Acumulada (Bs)',
                    template='plotly_white',
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    legend=dict(title='Indicadores', x=0.01, y=1.15, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos válidos para mostrar el gráfico.")
        elif 'TARIFA PLAN' not in df.columns:
            st.warning("⚠️ La columna 'TARIFA PLAN' no existe en el dataset.")
        else:
            st.warning("⚠️ Problema con las fechas: verifique la columna 'FECHA_TS'.")

    # 🔹 Filtros en el lienzo principal (usando st.expander)
    with st.expander("🧩 Monto recuperado filtrado por fecha del envio de la observacion", expanded=False):  # expanded=False para que esté colapsado por defecto
        #st.header("Filtros")

        # Selector de tipo de gráfico
        tipo_grafico = st.radio("Selecciona el tipo de gráfico:", ["Línea", "Barras"], horizontal=True,key="boton1")

        # Calcular el monto recuperado
        df['MONTO RECUPERADO'] = df.loc[df["PAGO DESPUES DE LA OBSERVACION"] == "SI", "TARIFA PLAN"]

        # Verificar si las columnas necesarias existen
        if 'MONTO RECUPERADO' in df.columns and 'FECHA_TS' in df.columns:
            # Convertir a tipo numérico y limpiar datos
            df['MONTO RECUPERADO'] = pd.to_numeric(df['MONTO RECUPERADO'], errors='coerce')
            df['FECHA_TS'] = pd.to_datetime(df['FECHA_TS'], errors='coerce')
            df_valid = df.dropna(subset=['MONTO RECUPERADO', 'FECHA_TS'])
            
            if not df_valid.empty:
                # Agrupar y sumar por fecha
                df_sum = df_valid.groupby('FECHA_TS', as_index=False)['MONTO RECUPERADO'].sum()
                
                # Crear gráfico
                fig = go.Figure()
                
                if tipo_grafico == "Línea":
                    # Línea principal
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['MONTO RECUPERADO'], 
                        mode='lines+markers',
                        name='Suma Monto Recuperado',
                        line=dict(color='#04514f', width=6),
                        marker=dict(size=8, color='darkblue', line=dict(width=1, color='white'))
                    ))

                    # Línea de tendencia (Media móvil de 7 días)
                    df_sum['Tendencia'] = df_sum['MONTO RECUPERADO'].rolling(window=7, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['Tendencia'], 
                        mode='lines',
                        name='Tendencia (Media 7 días)',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                
                else:
                    # Agregar un selector de color
                    color_barras = st.color_picker("Elige el color de las barras", "#04514f", key="color1")

                    fig.add_trace(go.Bar(
                        x=df_sum['FECHA_TS'], 
                        y=df_sum['MONTO RECUPERADO'], 
                        name='Suma Monto Recuperado',
                        marker=dict(color=color_barras)  # Aplica el color elegido
                    ))

                # Configuración del diseño
                fig.update_layout(
                    title='📊 Monto Recuperado por Fecha de Envío',
                    xaxis_title='Fecha de Envío',
                    yaxis_title='Monto Recuperado (Bs)',
                    template='plotly_white',
                    hovermode='x unified',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=-45),
                    yaxis=dict(showgrid=True, gridcolor='lightgray'),
                    legend=dict(title='Indicadores', x=0.01, y=1.15, orientation='h')
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No hay datos válidos para mostrar el gráfico.")
        elif 'MONTO RECUPERADO' not in df.columns:
            st.warning("⚠️ La columna 'MONTO RECUPERADO' no existe en el dataset.")
        else:
            st.warning("⚠️ Problema con las fechas: verifique la columna 'FECHA_TS'.")

        st.markdown(
            '<p style="background-color:#00111c; color:white; padding:10px; border-radius:5px;">'
            '📊 <b>Área de reporte</b>'
            '</p>',
            unsafe_allow_html=True
        )
    with st.expander("🧩 Descargar Reporte", expanded=False):  # expanded=False para que esté colapsado por defecto
    # 🔹 Título atractivo para la sección
        #st.title("🔄 Automatizar Reporte")

        # Agregar un toque visual con un subtítulo y estilo
        st.markdown("""
            <style>
            .reportview-container {
                background-color: #f4f6f9;
                font-family: 'Arial', sans-serif;
            }
            .sidebar .sidebar-content {
                background-color: #2E4053;
            }
            .css-1d391kg {
                color: #2D3436;
            }
            </style>
            """, unsafe_allow_html=True)
        st.subheader("Filtros de datos y opciones de descarga")

        # Mostrar el DataFrame filtrado con st.dataframe
        st.dataframe(df)

        # 🔹 Opción para descargar el archivo filtrado
    # 🔹 Opción para descargar el archivo filtrado automáticamente
    @st.cache_data
    def to_excel(df):
        """Convierte el DataFrame a un archivo Excel en formato bytes."""
        towrite = BytesIO()
        with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Datos Filtrados")
            writer._save()  # Guardar el archivo Excel
        return towrite.getvalue()

    # Convertir el DataFrame filtrado a Excel
    excel_file = to_excel(df)

    # Descargar automáticamente el archivo Excel
    st.download_button(
        label="📥 Descargar archivo Excel",
        data=excel_file,
        file_name="reporte_filtrado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

