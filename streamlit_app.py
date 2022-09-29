import altair as alt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import streamlit as st
import pandas as pd
from plotnine import *
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

global numeric_columns
global non_numeric_columns

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    res = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return res

alt.themes.enable("streamlit")

st.set_page_config(
    page_title="Data Analysis", page_icon="📊", layout="wide"
)

#*************************** Start Body section ***************************
st.header("Data Vizualization Tool")
st.subheader("Upload your CSV that you want to analyse")
uploaded_data = st.file_uploader(
    "Drag and Drop or Click to Upload", type=".csv", accept_multiple_files=False
)

if uploaded_data is None:
    st.info("Upload a file above to use your own data!")

else:
    st.success("Uploaded your file!")
    df = pd.read_csv(uploaded_data)

    # *************************** Start Sidebar section ***************************
    st.sidebar.subheader("Type of Analysis")
    selection = st.sidebar.selectbox(
        "Choose Analysis Type:", options=["Describe","Correlation", "Distribution", "Comparative Analysis"])

    if selection == "Distribution":
        df_col = list(df.columns)
        col_selections = st.sidebar.selectbox("Select Column", options=df_col)

    if selection == "Comparative Analysis":
        # add a select widget to the side bar
        chart_choice = st.sidebar.radio("Select the plot:", ["Histogram", "Boxplot", "Dotplot", "QQplot", "Scatterplot", "Violinplot", "Smoothed Conditional Mean"])

    # *************************** End Sidebar section ***************************

    with st.expander("Raw Dataframe"):
        st.dataframe(df)  # Same as st.write(df)


    with st.expander("Data Pivotin & Filtering"):
        df1 = df.copy()
        df1.columns = ['Name', 'Sex', 'Age', 'Height', 'Weight']
        # df = clean_data(df)
        # st.write(df1)
        res = aggrid_interactive_table(df=df)

        if res:
            st.write("You selected:")
            st.json(res["selected_rows"])

    if selection == "Distribution":

        if df[col_selections].dtype.kind in 'biufc': #check if feature is numeric

            col1, col2 = st.columns(2)

            col1.subheader("Histogram Analysis")

            #Seaborn like
            fig = Figure(figsize=(10, 5))
            ax = fig.subplots()
            sns.histplot(df[col_selections],
                         # kde_kws={"clip": (0.0, 2020)},
                         ax=ax,
                         kde=True,
            )
            ax.set_title(f"{col_selections} Distribution")
            #matplotlib like
            # fig = plt.figure(figsize=(10, 5))
            # plt.hist(df[col_selections])
            # plt.title(f"{col_selections} Distribution")

            col1.pyplot(fig)

            col2.subheader("Box-Plot Analysis")

            fig = plt.figure(figsize=(10, 5))
            sns.boxplot(df[col_selections])
            plt.title(f"{col_selections}")

            col2.pyplot(fig)

            st.subheader("ECDF Analysis")

            fig = plt.figure(figsize=(10, 5))
            sns.ecdfplot(data=df, x=col_selections)
            plt.title(f"{col_selections} ECDF")

            st.pyplot(fig)
        else:
            st.subheader("Histogram Analysis")

            fig = plt.figure(figsize=(10, 5))
            plt.hist(df[col_selections])
            plt.title(f"{col_selections} Distribution")

            st.pyplot(fig)

    elif selection == "Correlation":
        st.subheader("Feature Correlation Analysis")

        numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
        corr = df[numeric_columns].corr()
        fig = plt.figure(figsize=(5, 5))

        sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap="coolwarm",
            square=True
        )

        st.pyplot(fig)
    elif selection == "Describe":

        col1, col2, col3 = st.columns(3)

        col1.subheader("Basic information")
        # Basic information
        col1.write(df.info(verbose=True))

        # Describe the data
        col2.subheader("Describe the data")
        col2.write(df.describe())

        col3.subheader("Null Values")
        col3.write(df.isnull().sum())

    elif selection == "Comparative Analysis":
        st.subheader("Comparative Analysis")

        try:
            numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            non_numeric_columns.append(None)
        except Exception as e:
            print(e)
            st.write("Please upload file to the application.")

        p = ggplot(df)

        if chart_choice == "Histogram":

            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)
            bins = st.slider("Number of Bins", min_value=1, max_value=50, value=7)
            if cv != None:
                p = p + geom_histogram(aes(x=x, fill=cv, color=cv), position="identity", alpha=.4, bins=bins)
            else:
                p = p + geom_histogram(aes(x=x), color="darkblue", fill="lightblue", bins=bins)

        if chart_choice == "Boxplot":
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)

            if cv != None:
                p = p + geom_boxplot(aes(x=cv, y=x, fill=cv), notch = True) + coord_flip()
            else:
                p = p + geom_boxplot(aes(x=1, y=x, width=.1, notch = True), color="darkblue", fill="lightblue") + coord_flip()

        if chart_choice == "Dotplot":
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)
            if cv != None:
                p = p + geom_jitter(aes(x=cv, y=x, fill=cv, color=cv), size=2, height=0, width=.1) + coord_flip()
            else:
                p = p + geom_jitter(aes(x=1, y=x), size=2, height=0, width=.1) + coord_flip()

        if chart_choice == "QQplot":
            x = st.selectbox('X-Axis', options=numeric_columns)
            cv = st.selectbox("Color", options=non_numeric_columns)
            if cv != None:
                p = p + stat_qq(aes(sample=x, color=cv)) + stat_qq_line(aes(sample=x, color=cv)) + labs(
                    x="Theoretical Quantiles", y="Sample Quantiles")
            else:
                p = p + stat_qq(aes(sample=x)) + stat_qq_line(aes(sample=x)) + labs(x="Theoretical Quantiles",
                                                                                    y="Sample Quantiles")

        if chart_choice == "Scatterplot":
            x = st.selectbox('X-Axis', options=numeric_columns)
            y = st.selectbox('Y-Axis', options=numeric_columns, index=1)
            cv = st.selectbox("Color", options=non_numeric_columns)
            if cv != None:
                p = p + geom_point(aes(x=x, y=y, color=cv))
            else:
                p = p + geom_point(aes(x=x, y=y))

        if chart_choice == "Violinplot":
            x = st.selectbox('X-Axis', options=numeric_columns)
            y = st.selectbox('Y-Axis', options=numeric_columns, index=1)
            cv = st.selectbox("Color", options=non_numeric_columns)
            if cv != None:
                p = p + geom_violin(aes(x=x, y=y, color=cv))
            else:
                p = p + geom_violin(aes(x=x, y=y))

        if chart_choice == "Smoothed Conditional Mean":
            x = st.selectbox('X-Axis', options=numeric_columns)
            y = st.selectbox('Y-Axis', options=numeric_columns, index=1)
            cv = st.selectbox("Color", options=non_numeric_columns)
            span = st.slider("Span smoother", min_value=0.0, max_value=1.0, value=0.3)
            if cv != None:
                p = ggplot(df, aes(x=x, y=y, color=cv)) \
                    + geom_point() \
                    + geom_smooth(span=span)
            else:
                p = ggplot(df, aes(x=x, y=y)) \
                    + geom_point() \
                    + geom_smooth(span=span)


        st.pyplot(ggplot.draw(p))
#*************************** End Body section ***************************
