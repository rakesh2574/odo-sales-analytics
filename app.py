"""
Sales Analytics Platform - Proper Enhanced Version
Uses schema for validation, relationships for knowledge graph
Validates BEFORE execution, not after
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
import sqlite3
import os
import re
from typing import Dict, List, Optional, Tuple, Set

# Page config
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide"
)

# Minimal CSS
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
        st.error("Database not found. Please ensure sales_order.db is in the same folder.")
        return None
    return sqlite3.connect(db_file, check_same_thread=False)


@st.cache_data
def load_schema():
    """Load database schema"""
    try:
        with open('database_schema.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading schema: {e}")
        return None


def get_valid_columns_and_tables(schema: dict) -> Tuple[Set[str], Set[str], Dict[str, List[str]]]:
    """
    Extract all valid column names, table names, and table-column mappings from schema
    """
    valid_tables = set()
    valid_columns = set()
    table_columns = {}

    for table in schema['tables']:
        table_name = table['table_name']
        valid_tables.add(table_name)
        table_columns[table_name] = []

        for col in table['columns']:
            col_name = col['name']
            valid_columns.add(col_name)  # Just the column name
            table_columns[table_name].append(col_name)

    return valid_columns, valid_tables, table_columns


def validate_sql_against_schema(sql: str, schema: dict) -> Tuple[bool, Optional[str]]:
    """
    PROPER validation: Check that columns exist in schema
    Handles table aliases correctly
    """
    valid_columns, valid_tables, table_columns = get_valid_columns_and_tables(schema)

    # SQL keywords that are not columns
    sql_keywords = {
        'select', 'from', 'where', 'group', 'order', 'limit', 'offset',
        'join', 'inner', 'left', 'right', 'outer', 'on', 'as', 'and', 'or',
        'having', 'distinct', 'count', 'sum', 'avg', 'max', 'min', 'case',
        'when', 'then', 'else', 'end', 'between', 'like', 'in', 'not', 'is',
        'null', 'asc', 'desc', 'by', 'all', 'exists', 'union', 'intersect'
    }

    # Extract all table.column references (e.g., "so.id", "c.name")
    column_refs = re.findall(r'\b(\w+)\.(\w+)\b', sql.lower())

    # Validate each column reference
    for table_alias, col_name in column_refs:
        # Skip if it's a SQL keyword
        if col_name in sql_keywords:
            continue

        # Check if column exists in ANY table (we don't care about the alias)
        if col_name not in valid_columns:
            # Not a valid column - find similar columns for helpful error
            similar = [c for c in valid_columns if col_name in c or c in col_name]
            error_msg = f"Column '{col_name}' does not exist in schema."
            if similar:
                error_msg += f" Did you mean: {', '.join(similar[:3])}?"
            return False, error_msg

    # Also check standalone column names in SELECT clause (without table prefix)
    # Extract SELECT clause
    select_match = re.search(r'select\s+(.*?)\s+from', sql.lower(), re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        # Remove aggregations and functions
        select_clause = re.sub(r'(sum|avg|count|max|min|distinct)\s*\([^)]*\)', '', select_clause)
        # Extract potential column names (word before 'as' or just words)
        potential_cols = re.findall(r'\b([a-z_][a-z0-9_]*)\b', select_clause)

        for col in potential_cols:
            # Skip SQL keywords, numbers, and already validated table.column refs
            if col in sql_keywords or col.isdigit():
                continue
            # Skip if it's a table name (could be alias)
            if col in valid_tables:
                continue
            # Check if it's a valid column
            if col not in valid_columns:
                # Could be an alias or aggregation result - check if it appears in SELECT with AS
                if f' as {col}' in sql.lower() or f' {col}' in sql.lower():
                    continue  # It's an alias for a result
                # Otherwise, it might be an error
                # But to avoid false positives, only flag if it looks like a real column attempt
                if '_' in col or len(col) > 3:  # Likely meant to be a column
                    return False, f"Column '{col}' not found in schema. Check spelling."

    return True, None


def is_business_query(question: str) -> bool:
    """Check if query is related to sales/business data"""
    business_keywords = [
        'order', 'sale', 'customer', 'product', 'revenue', 'price', 'team',
        'salesperson', 'contact', 'purchase', 'invoice', 'payment', 'quantity',
        'total', 'average', 'top', 'best', 'worst', 'most', 'least', 'show',
        'list', 'find', 'what', 'how many', 'which', 'who', 'when', 'each'
    ]

    question_lower = question.lower()
    return any(keyword in question_lower for keyword in business_keywords)


def validate_query(question: str) -> Tuple[bool, Optional[str]]:
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


def get_example_queries() -> List[Dict[str, str]]:
    """
    Few-shot examples for SQL generation - CRITICAL for accuracy
    These are REAL working queries from your client
    """
    return [
        {
            "question": "Show all orders from LightsUp",
            "sql": """select so.id, so.name as order_ref, c.name as customer
from sale_order so
join contact c on so.partner_id = c.id
where c.name like '%LightsUp%'"""
        },
        {
            "question": "What are the top 5 products by revenue?",
            "sql": """select pv.name as product, sum(sol.price_total) as total_revenue
from sale_order_line sol
join product_variant pv on sol.product_id = pv.id
group by pv.name
order by total_revenue desc
limit 5"""
        },
        {
            "question": "List the top 5 selling products by quantity in a date range",
            "sql": """select 
    pv.name as product,
    sum(sol.product_uom_qty) as total_quantity_sold
from sale_order_line sol
join product_variant pv on pv.id = sol.product_id
join sale_order so on so.id = sol.order_id
where so.date_order between '2025-01-01' and '2025-12-31'
group by pv.name
order by total_quantity_sold desc
limit 5"""
        },
        {
            "question": "For each customer, find the product they purchased most frequently",
            "sql": """select 
    c.name as customer,
    pv.name as product,
    sum(sol.product_uom_qty) as total_quantity
from sale_order_line sol
join sale_order so on so.id = sol.order_id
join contact c on c.id = so.partner_id
join product_variant pv on pv.id = sol.product_id
group by c.name, pv.name
having sum(sol.product_uom_qty) = (
    select max(total_qty)
    from (
        select 
            sum(sol2.product_uom_qty) as total_qty
        from sale_order_line sol2
        join sale_order so2 on so2.id = sol2.order_id
        where so2.partner_id = so.partner_id
        group by sol2.product_id
    ) sub
)"""
        },
        {
            "question": "Show total sales by customer",
            "sql": """select c.name as customer, sum(sol.price_total) as total_sales
from sale_order_line sol
join contact c on sol.order_partner_id = c.id
group by c.name
order by total_sales desc"""
        },
        {
            "question": "Which salesperson has the most orders?",
            "sql": """select u.name as salesperson, count(distinct so.id) as order_count
from sale_order so
join user u on so.user_id = u.id
group by u.name
order by order_count desc
limit 1"""
        },
        {
            "question": "What is the average order value?",
            "sql": """select avg(order_total) as avg_order_value
from (
  select order_id, sum(price_total) as order_total
  from sale_order_line
  group by order_id
)"""
        },
        {
            "question": "Show total revenue by sales team",
            "sql": """select st.name as team, sum(sol.price_total) as total_revenue
from sale_order so
join sales_team st on so.team_id = st.id
join sale_order_line sol on so.id = sol.order_id
group by st.name
order by total_revenue desc"""
        },
        {
            "question": "Who are the top 10 customers by revenue?",
            "sql": """select c.name as customer, sum(sol.price_total) as total_revenue
from sale_order_line sol
join contact c on sol.order_partner_id = c.id
group by c.name
order by total_revenue desc
limit 10"""
        }
    ]


def create_enhanced_schema_prompt(schema: dict) -> str:
    """
    Create enhanced prompt with complete schema information
    """
    prompt = """You are a SQL expert. Convert natural language to SQLite queries.

⚠️ CRITICAL RULES - FOLLOW EXACTLY:
1. Use ONLY column names from the schema below - NO other columns exist
2. Use ONLY table names from the schema below - NO other tables exist
3. Return ONLY the SQL query - no explanations, no markdown, no ```
4. Use lowercase for SQL keywords
5. Do NOT add semicolons
6. Use table aliases (e.g., so for sale_order)

📋 DATABASE SCHEMA - THESE ARE THE ONLY VALID TABLES AND COLUMNS:
"""

    # Add detailed schema
    for table in schema['tables']:
        prompt += f"\n🔹 Table: {table['table_name']}"
        if 'description' in table:
            prompt += f" - {table['description']}"
        prompt += "\n   Columns (ONLY these columns exist):\n"

        for col in table['columns']:
            col_info = f"   • {col['name']} ({col['type']})"
            if 'description' in col:
                col_info += f" - {col['description']}"
            if col.get('foreign_key'):
                col_info += f" [FK → {col['foreign_key']}]"
            prompt += col_info + "\n"

    # Add relationships
    prompt += "\n🔗 TABLE RELATIONSHIPS (use these for JOINs):\n"
    for table in schema['tables']:
        if 'relationships' in table:
            for rel in table['relationships']:
                fk = rel['foreign_key']
                ref = rel['references']
                prompt += f"   • {table['table_name']}.{fk} = {ref}\n"
                prompt += f"     JOIN: join {ref.split('.')[0]} on {table['table_name']}.{fk} = {ref}\n"

    # Add examples
    prompt += "\n✅ EXAMPLE QUERIES (follow these patterns):\n"
    examples = get_example_queries()
    for i, ex in enumerate(examples[:7], 1):
        prompt += f"\n{i}. Q: {ex['question']}\n"
        prompt += f"   SQL: {ex['sql']}\n"

    prompt += "\n⚠️ VALIDATION CHECKLIST:"
    prompt += "\n   ✓ Every column in my query exists in the schema above"
    prompt += "\n   ✓ Every table in my query exists in the schema above"
    prompt += "\n   ✓ My JOINs use the exact foreign key relationships listed above"
    prompt += "\n   ✓ I'm using lowercase keywords"
    prompt += "\n   ✓ I'm not inventing any column or table names"

    prompt += "\n\n📝 USER QUESTION: "

    return prompt


def generate_sql_with_validation(question: str, schema: dict, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate SQL and validate BEFORE execution
    Retry with feedback if validation fails
    """
    try:
        client = OpenAI(api_key=api_key)
        prompt = create_enhanced_schema_prompt(schema) + question

        # Try up to 2 times
        for attempt in range(2):
            if attempt > 0:
                prompt += "\n\n⚠️ Your previous SQL had errors. Please fix and regenerate using ONLY the columns from the schema."

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a SQL expert. Generate ONLY valid SQL using the exact column and table names provided in the schema. Never invent column names."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_tokens=500
            )

            sql = response.choices[0].message.content.strip()

            # Clean up
            if sql.startswith('```'):
                lines = sql.split('\n')
                sql = '\n'.join([l for l in lines if not l.strip().startswith('```')])

            if sql.lower().strip().startswith('sql'):
                sql = sql[3:].strip()

            sql = sql.strip().rstrip(';')

            # VALIDATE BEFORE EXECUTION
            is_valid, error = validate_sql_against_schema(sql, schema)

            if is_valid:
                return sql, None  # Success!
            else:
                # Add error feedback for retry
                prompt += f"\n\n❌ Error: {error}"
                if attempt == 1:  # Last attempt
                    return None, f"Validation failed: {error}"

        return None, "Could not generate valid SQL after 2 attempts"

    except Exception as e:
        return None, f"Error: {str(e)}"


def execute_query(sql: str, conn) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Execute SQL query - only called AFTER validation passes"""
    try:
        return pd.read_sql_query(sql, conn), None
    except Exception as e:
        return None, f"Execution error: {str(e)}"


def generate_natural_language_answer(question: str, df: pd.DataFrame, api_key: str) -> str:
    """Generate natural language answer from query results"""
    try:
        client = OpenAI(api_key=api_key)

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

Provide a clear answer in 2-3 sentences. Use natural language. Format numbers with $ and commas where appropriate."""

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


def format_dataframe_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format dataframe for display"""
    display_df = df.copy()

    for col in display_df.columns:
        if 'date' in col.lower() and display_df[col].dtype in ['float64', 'float32']:
            try:
                display_df[col] = display_df[col].apply(
                    lambda x: excel_date_to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else None
                )
            except:
                pass
        elif any(x in col.lower() for x in ['revenue', 'total', 'price', 'sales']) and display_df[col].dtype in ['float64', 'float32']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else None)

    return display_df


def get_stats(conn):
    """Get database statistics"""
    try:
        cursor = conn.cursor()

        stats = {}
        for table in ['contact', 'sales_team', 'user', 'product_variant', 'sale_order', 'sale_order_line']:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            except:
                stats[table] = 0

        try:
            cursor.execute("SELECT SUM(price_total) FROM sale_order_line")
            result = cursor.fetchone()
            stats['revenue'] = result[0] if result[0] else 0
        except:
            stats['revenue'] = 0

        try:
            cursor.execute("SELECT AVG(order_total) FROM (SELECT order_id, SUM(price_total) as order_total FROM sale_order_line GROUP BY order_id)")
            result = cursor.fetchone()
            stats['avg_order'] = result[0] if result[0] else 0
        except:
            stats['avg_order'] = 0

        return stats
    except Exception as e:
        return {}


def main():
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""

    st.title("📊 Sales Analytics Platform")
    st.caption("Ask questions about your sales data in plain English")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")

        st.divider()
        st.header("📊 Database Overview")

        conn_stats = get_db_connection()
        if conn_stats:
            stats = get_stats(conn_stats)
            if stats:
                st.metric("Total Customers", f"{stats.get('contact', 0):,}")
                st.metric("Total Orders", f"{stats.get('sale_order', 0):,}")
                st.metric("Total Revenue", f"${stats.get('revenue', 0):,.0f}")



    schema = load_schema()
    conn = get_db_connection()

    if not schema or not conn:
        st.error("System initialization failed.")
        return

    st.subheader("Ask a Question")

    # Quick buttons
    st.write("**Quick Questions:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Top 5 Products", use_container_width=True):
            st.session_state.current_query = "What are the top 5 products by revenue?"
            st.rerun()

    with col2:
        if st.button("📈 Top by Quantity", use_container_width=True):
            st.session_state.current_query = "List top 5 products by quantity sold in 2025"
            st.rerun()

    with col3:
        if st.button("👥 Customer Preference", use_container_width=True):
            st.session_state.current_query = "For each customer, what product do they buy most?"
            st.rerun()

    with col4:
        if st.button("💰 Top Customers", use_container_width=True):
            st.session_state.current_query = "Who are the top 10 customers by revenue?"
            st.rerun()

    st.divider()

    query = st.text_area(
        "Or type your own question:",
        value=st.session_state.current_query,
        height=80,
        placeholder="e.g., Which products generated the most revenue?"
    )

    if query != st.session_state.current_query:
        st.session_state.current_query = query

    if st.button("▶️ Get Answer", type="primary"):
        if not api_key:
            st.error("⚠️ Please enter your OpenAI API key in the sidebar")
            return

        if not query or len(query.strip()) < 5:
            st.warning("⚠️ Please enter a question (at least 5 characters)")
            return

        is_valid, validation_msg = validate_query(query)
        if not is_valid:
            st.error(f"🚫 {validation_msg}")
            return

        # Generate and VALIDATE SQL
        with st.spinner("🤖 Processing your question..."):
            sql, error = generate_sql_with_validation(query, schema, api_key)

        if error:
            st.error(f"❌ {error}")
            st.info("💡 Try rephrasing your question or use one of the quick questions above")
            return

        # Execute (only if validation passed)
        with st.spinner("📊 Executing query..."):
            result_df, error = execute_query(sql, conn)

        if error:
            st.error(f"❌ {error}")
            with st.expander("🔍 View SQL"):
                st.code(sql, language='sql')
            return

        # Generate answer
        with st.spinner("✍️ Preparing answer..."):
            nl_answer = generate_natural_language_answer(query, result_df, api_key)

        st.success("✅ Query executed successfully!")

        st.subheader("💬 Answer")
        st.info(nl_answer)

        if result_df is not None and len(result_df) > 0:
            st.divider()
            st.subheader("📋 Data")

            display_df = format_dataframe_display(result_df)
            st.dataframe(display_df, use_container_width=True, height=400)

            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

            with st.expander("🔧 View SQL Query"):
                st.code(sql, language='sql')
        else:
            st.info("ℹ️ No data found")


if __name__ == "__main__":
    main()