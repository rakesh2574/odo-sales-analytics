"""
Sales Analytics Platform - Ultra Clean Client Version
Simple, readable, professional
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
import sqlite3
import os

# Page config
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Ultra minimal CSS - just hide branding
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1 {color: #2c3e50; text-align: center; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    """Get database connection"""
    db_file = 'sales_order.db'
    if not os.path.exists(db_file):
        st.error("Database not found. Please run: python create_database.py")
        return None
    return sqlite3.connect(db_file, check_same_thread=False)


@st.cache_data
def load_schema():
    """Load database schema"""
    try:
        with open('database_schema.json', 'r') as f:
            return json.load(f)
    except:
        return None


def is_business_query(question):
    """Check if query is related to sales/business data"""
    business_keywords = [
        'order', 'sale', 'customer', 'product', 'revenue', 'price', 'team',
        'salesperson', 'contact', 'purchase', 'invoice', 'payment', 'quantity',
        'total', 'average', 'top', 'best', 'worst', 'most', 'least', 'show',
        'list', 'find', 'what', 'how many', 'which', 'who', 'when'
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in business_keywords)


def validate_query(question):
    """Validate if query is appropriate for sales order data"""
    non_business_terms = [
        'president', 'weather', 'news', 'movie', 'sports', 'game',
        'celebrity', 'politics', 'capital', 'country', 'wikipedia'
    ]

    question_lower = question.lower()
    if any(term in question_lower for term in non_business_terms):
        return False, "This appears to be a general knowledge question. Please ask questions about sales orders, customers, products, or revenue."

    if not is_business_query(question):
        return False, "Please ask questions related to sales orders, customers, products, or business metrics."

    return True, None


def excel_date_to_datetime(excel_date):
    """Convert Excel serial date to datetime"""
    try:
        return datetime(1899, 12, 30) + pd.Timedelta(days=excel_date)
    except:
        return None


def create_schema_prompt(schema):
    """Create schema prompt for LLM"""
    prompt = """Convert natural language to SQL for a sales order database.

DATABASE SCHEMA:
"""

    for table in schema['tables']:
        prompt += f"\n{table['table_name']} - {table['description']}\n"
        for col in table['columns']:
            prompt += f"  {col['name']} ({col['type']}): {col['description']}\n"

    prompt += """
RULES:
- Return ONLY the SQL query, no explanations
- Use SQLite syntax
- No semicolons
- Use lowercase keywords

Query: """

    return prompt


def generate_sql(question, schema, api_key):
    """Generate SQL from natural language"""
    try:
        client = OpenAI(api_key=api_key)
        prompt = create_schema_prompt(schema) + question

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Return only SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        sql = response.choices[0].message.content.strip()

        # Clean up
        if sql.startswith('```'):
            sql = '\n'.join([l for l in sql.split('\n') if not l.startswith('```')])
            if sql.lower().startswith('sql'):
                sql = sql[3:]

        return sql.strip().rstrip(';'), None

    except Exception as e:
        return None, str(e)


def execute_query(sql, conn):
    """Execute SQL query"""
    try:
        return pd.read_sql_query(sql, conn), None
    except Exception as e:
        return None, str(e)


def generate_natural_language_answer(question, df, api_key):
    """Generate natural language answer from query results"""
    try:
        client = OpenAI(api_key=api_key)

        # Prepare data summary
        if df is not None and len(df) > 0:
            data_summary = f"Query returned {len(df)} rows.\n\n"
            data_summary += "Sample data:\n"
            for idx, row in df.head(10).iterrows():
                data_summary += f"Row {idx+1}: "
                parts = []
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, float):
                            if any(x in col.lower() for x in ['revenue', 'total', 'price', 'sales']):
                                val = f"${val:,.2f}"
                            else:
                                val = f"{val:,.2f}"
                        parts.append(f"{col}={val}")
                data_summary += ", ".join(parts) + "\n"
        else:
            data_summary = "No results found."

        prompt = f"""You are a business analyst. Answer the question clearly based on the data.

Question: {question}

Data:
{data_summary}

Provide a clear answer in 2-3 sentences. Use natural language. Format numbers with $ and commas."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst who explains data clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    except:
        if df is not None and len(df) > 0:
            return f"Found {len(df)} result(s)."
        return "No results found."


def format_dataframe_display(df):
    """Format dataframe for display"""
    display_df = df.copy()

    # Format dates
    for col in display_df.columns:
        if 'date' in col.lower() and display_df[col].dtype in ['float64', 'float32']:
            try:
                display_df[col] = display_df[col].apply(
                    lambda x: excel_date_to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else None
                )
            except:
                pass
        # Format currency
        elif any(x in col.lower() for x in ['revenue', 'total', 'price', 'sales']) and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else None)

    return display_df


def get_stats(conn):
    """Get database statistics"""
    cursor = conn.cursor()

    stats = {}
    for table in ['contact', 'sales_team', 'user', 'product_variant', 'sale_order', 'sale_order_line']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[table] = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(price_total) FROM sale_order_line")
    stats['revenue'] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(order_total) FROM (SELECT order_id, SUM(price_total) as order_total FROM sale_order_line GROUP BY order_id)")
    stats['avg_order'] = cursor.fetchone()[0]

    return stats


def main():
    # Initialize session state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    # Header
    st.title("📊 Sales Analytics Platform")
    st.caption("Ask questions about your sales data in plain English")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")

        st.divider()
        st.header("📊 Database Overview")

        conn_stats = get_db_connection()
        if conn_stats:
            stats = get_stats(conn_stats)
            st.metric("Total Customers", f"{stats['contact']}")
            st.metric("Total Orders", f"{stats['sale_order']}")
            st.metric("Total Products", f"{stats['product_variant']}")
            st.metric("Total Revenue", f"${stats['revenue']:,.0f}")

    # Load resources
    schema = load_schema()
    conn = get_db_connection()

    if not schema or not conn:
        st.error("System initialization failed.")
        return

    # Main interface
    st.subheader("Ask a Question")

    # Quick query buttons
    st.write("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Top 5 Products", use_container_width=True):
            st.session_state.current_query = "What are the top 5 products by revenue?"
            st.rerun()
        if st.button("👥 Sales by Team", use_container_width=True):
            st.session_state.current_query = "Show total sales by sales team"
            st.rerun()

    with col2:
        if st.button("🛒 Total Orders", use_container_width=True):
            st.session_state.current_query = "How many orders do we have?"
            st.rerun()
        if st.button("💰 Average Order", use_container_width=True):
            st.session_state.current_query = "What is the average order value?"
            st.rerun()

    with col3:
        if st.button("🏆 Top Salesperson", use_container_width=True):
            st.session_state.current_query = "Which salesperson has the most orders?"
            st.rerun()
        if st.button("📈 Customer Sales", use_container_width=True):
            st.session_state.current_query = "Show total sales by customer"
            st.rerun()

    with col4:
        if st.button("🔍 LightsUp Orders", use_container_width=True):
            st.session_state.current_query = "Show all orders from LightsUp"
            st.rerun()
        if st.button("💼 Top 10 Customers", use_container_width=True):
            st.session_state.current_query = "Who are the top 10 customers by revenue?"
            st.rerun()

    st.divider()

    # Query input
    query = st.text_area(
        "Or type your own question:",
        value=st.session_state.current_query,
        height=80,
        placeholder="e.g., Which products generated the most revenue?"
    )

    # Update session state
    if query != st.session_state.current_query:
        st.session_state.current_query = query

    # Execute button
    execute_btn = st.button("▶️ Get Answer", type="primary")

    # Execute query
    if execute_btn:
        if not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar")
            return

        if not query or len(query.strip()) < 5:
            st.warning("⚠️ Please enter a question (at least 5 characters)")
            return

        # Validate query
        is_valid, validation_msg = validate_query(query)
        if not is_valid:
            st.error(f"🚫 {validation_msg}")
            st.info("💡 Try asking about: sales orders, customers, products, revenue, or sales teams")
            return

        # Generate SQL
        with st.spinner("🤖 Processing..."):
            sql, error = generate_sql(query, schema, api_key)

        if error:
            st.error(f"❌ Error: {error}")
            return

        # Execute SQL
        with st.spinner("📊 Analyzing..."):
            result_df, error = execute_query(sql, conn)

        if error:
            st.error(f"❌ Query failed: {error}")
            return

        # Generate answer
        with st.spinner("✍️ Preparing answer..."):
            nl_answer = generate_natural_language_answer(query, result_df, api_key)

        # Show answer - SIMPLE AND READABLE
        st.subheader("💬 Answer")
        st.info(nl_answer)  # Using st.info for simple, readable display

        # Show data table
        if result_df is not None and len(result_df) > 0:
            st.divider()
            st.subheader("📋 Data")

            display_df = format_dataframe_display(result_df)
            st.dataframe(display_df, use_container_width=True, height=400)

            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

            # SQL in expander
            with st.expander("🔧 View Technical Details (SQL Query)"):
                st.code(sql, language='sql')
                st.caption("SQL query executed on the database")
        else:
            st.info("ℹ️ No data found")


if __name__ == "__main__":
    main()