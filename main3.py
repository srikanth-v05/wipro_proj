import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
import json
import seaborn as sns
import plotly.express as px
import os

# Streamlit page configuration
st.set_page_config(page_title="Wipro Project", layout="wide")

# Configure the Gemini API key
api_key = st.secrets["api_key"]
api_key = os.getenv("api_key", api_key)

# Setup the Gemini API configuration
genai.configure(api_key=api_key)

def shorten_subject_name(subject):
    return ''.join([word[0] for word in subject.split()])

# Function to load and process data
def load_and_process_data(uploaded_file):
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)

        # Round and convert 'mark' to integer
        if 'mark' in data.columns:
            data['mark'] = data['mark'].round(0).astype(int)

        # Rename columns for better readability
        column_rename_mapping = {
            'course_name': 'Course Name',
            'course_id': 'Course ID',
            'attempt': 'Attempt ID',
            'candidate_name': 'Candidate Name',
            'candidate_email': 'Candidate Email',
            'mark': 'Marks',
            'grade': 'Grade',
            'Performance_category': 'Performance Category'
        }
        data.rename(columns=column_rename_mapping, inplace=True)

        st.session_state["processed_data"] = data
        st.success("File processed successfully!")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

# Function to generate To-Do lists for all courses
def generate_todo_list_all_courses(courses, syllabus_dict):
    combined_todo_list = {}
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    for course in courses:
        if course in syllabus_dict:
            syllabus_text = syllabus_dict[course]

            prompt = f"""
            Course Name: {course}
            Below is the syllabus for this course:

            {syllabus_text}

            Based on the syllabus, generate a To-Do list to help students prepare for the course. 
            Limit the list to a maximum of 3 topics for each level:
            - 'Weak': Give me easier topic to gain knowleadge for basics
            - 'Medium': Intermediate topics for further learning.
            - 'Strong': Advanced topics for in-depth understanding.

            Format the To-Do list like this:
            {{
                '{course}': {{
                    'Weak': ['topic 1', 'topic 2', 'topic 3'],
                    'Medium': ['topic 1', 'topic 2', 'topic 3'],
                    'Strong': ['topic 1', 'topic 2', 'topic 3'],
                }}
            }}
            """
            try:
                response = model.generate_content([prompt])
                response_text = response.text

                # Extract JSON from response
                start_index = response_text.find("{")
                end_index = response_text.rfind("}") + 1
                course_todo_list = json.loads(response_text[start_index:end_index])

                # Merge with the combined To-Do list
                combined_todo_list.update(course_todo_list)

            except Exception as e:
                st.error(f"Error generating To-Do list for course: {course}. Details: {e}")
                continue

    return combined_todo_list

# Sidebar with option menu
with st.sidebar:
    selected_phase = option_menu(
        menu_title="Select Phase",
        options=["Anar1", "Anar2", "Anar3", "Anar4", "Anar5", "Anar6"],
        icons=["1-circle", "2-circle", "3-circle", "4-circle", "5-circle", "6-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Initialize session state for processed data
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "data_with_todo" not in st.session_state:
    st.session_state["data_with_todo"] = None
                            
# Content for each phase
if selected_phase == "Anar1":
    st.title("Phase 1: Anar1")
    st.subheader("Read performance report from multiple format (CSV data and excel data) ")
    st.write("Upload a performance report file to process data for Anar1.")

    # File uploader for Anar1
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Processing the file..."):
            load_and_process_data(uploaded_file)

    # Display the processed data
    if st.session_state["processed_data"] is not None:
        st.subheader("Processed Data")
        st.dataframe(st.session_state["processed_data"])

elif selected_phase == "Anar2":
    st.title("Phase 2: Anar2")
    st.subheader("Highlight the Strength, Weakness of candidates")
    st.write("Highlight the Strength and Weakness of candidates.")

    # Check if processed data is available
    if st.session_state["processed_data"] is not None:
        data = st.session_state["processed_data"]
        

        # Step 1: Distribution Analysis
        st.subheader("Distribution Analysis")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        sns.histplot(data['Marks'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Marks')

        sns.countplot(x='Grade', data=data, order=sorted(data['Grade'].unique()), ax=axes[0, 1])
        axes[0, 1].set_title('Grade Distribution')

        sns.countplot(x='Performance Category', data=data, order=['Poor', 'Medium', 'High'], ax=axes[1, 0])
        axes[1, 0].set_title('Performance Distribution')

        # Correlation Heatmap
        label_encoder_grade = LabelEncoder()
        label_encoder_performance = LabelEncoder()
        data['grade_encoded'] = label_encoder_grade.fit_transform(data['Grade'])
        data['performance_encoded'] = label_encoder_performance.fit_transform(data['Performance Category'])

        correlation_matrix = data[['Marks', 'grade_encoded', 'performance_encoded']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Matrix')

        plt.tight_layout()
        st.pyplot(fig)

        # Step 2: Trend Analysis
        st.subheader("Trend Analysis")
        mark_by_attempt = data.groupby('Attempt ID')['Marks'].mean()
        st.write("Average Marks by Attempt:")
        st.dataframe(mark_by_attempt)

        # Step 3: Model Training and Evaluation
        st.subheader("Model Training and Evaluation")

        # Encode 'course_id' for model training
        label_encoder_course = LabelEncoder()
        data['course_id_encoded'] = label_encoder_course.fit_transform(data['Course ID'])

        # Define features and target (exclude Candidate Name and Course Name)
        X = data[['Marks', 'Attempt ID', 'course_id_encoded']]
        y = label_encoder_performance.fit_transform(data['Performance Category'])

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest Classifier
        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred_forest = random_forest.predict(X_test)

        # Add Candidate Name and Course Name to X_test
        X_test['Candidate Name'] = data.loc[X_test.index, 'Candidate Name']
        X_test['Course Name'] = data.loc[X_test.index, 'Course Name']
        X_test['Course id'] = label_encoder_course.inverse_transform(data.loc[X_test.index, 'course_id_encoded'])
        X_test['Predicted Performance'] = label_encoder_performance.inverse_transform(y_pred_forest)

        # Display the updated DataFrame
        st.write("Test features along with predictions, including Candidate Name and Course Name:")
        st.dataframe(X_test[['Candidate Name', 'Course Name','Course id', 'Marks', 'Attempt ID', 'Predicted Performance']])

        # Save updated X_test to CSV
        X_test.to_csv('X_test_with_predictions.csv', index=False)
        # Create a new DataFrame to store the categorized subjects for each candidate
        subject_categories = {
            'Candidate Name': [],
            'High': [],
            'Medium': [],
            'Poor': []
        }
        # Filter the dataset to keep only the highest attempt per candidate per course
        filtered_data_coursewise = data.loc[data.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmax()]

        # Sort the data for better clarity (optional)
        filtered_data_coursewise = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)
        # Iterate through the X_test to categorize subjects
        for _, row in filtered_data_coursewise.iterrows():
            candidate_name = row['Candidate Name']
            predicted_performance = row['Performance Category']
            course_name = row['Course Name']

            # Add the course to the corresponding category
            if predicted_performance == 'High':
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append(course_name)
                subject_categories['Medium'].append('')
                subject_categories['Poor'].append('')
            elif predicted_performance == 'Medium':
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append('')
                subject_categories['Medium'].append(course_name)
                subject_categories['Poor'].append('')
            else:
                subject_categories['Candidate Name'].append(candidate_name)
                subject_categories['High'].append('')
                subject_categories['Medium'].append('')
                subject_categories['Poor'].append(course_name)

        # Create a DataFrame from the dictionary
        subject_category_df = pd.DataFrame(subject_categories)

        # Remove any duplicates, so that each candidate only appears once
        subject_category_df = subject_category_df.groupby('Candidate Name').agg(lambda x: ', '.join(filter(None, x))).reset_index()

        # Display the new table
        st.write("Candidate Performance Categories by Subject:")
        st.dataframe(subject_category_df)

        # Save the new table to CSV
        subject_category_df.to_csv('candidate_performance_by_subject.csv', index=False)
        st.success("Candidate performance by subject saved to 'candidate_performance_by_subject.csv'.")
    else:
        st.warning("Please upload and process data in Anar1 first.")

# Anar3 Phase Logic
elif selected_phase == "Anar3":
    st.title("Phase 3: Anar3")
    st.subheader("Recommend the To Do Course list for Weak Performers")
    st.write("Generate To-Do Lists for All Courses and Map Them to Students.")

    # Ensure processed data is available
    if st.session_state["processed_data"] is not None:
        data = st.session_state["processed_data"].copy()
        filtered_data_coursewise = data.loc[data.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmin()]
        # Sort the data for better clarity (optional)
        data = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)
        courses = data['Course Name'].unique()      
        # Predefined syllabus for each course
        syllabus_dict = {
            "Engineering Mathematics": """
UNIT I MULTIPLE INTEGRALS

(12 Hrs)

Multiple Integrals, change of order of integration and change of variables in double integrals (Cartesian to polar). Applications: Areas by double integration and volumes by triple integration (Cartesian and polar).

(12 Hrs)

UNIT II LAPLACE TRANSFORMS AND INVERSE LAPLACE TRANSFORMS

Definition, Transforms of elementary functions, properties. Transform of derivatives and integrals. Multiplication by t and division by t. Transform of unit step function, transform of periodic functions. Initial and final value theorems, Methods for determining inverse Laplace Transforms, Convolution theorem, Application to differential equations and integral equations. Evaluation of integrals by Laplace transforms.

UNIT III FOURIER SERIES

(12 Hrs)

Dirichlet's conditions - General Fourier series - Expansion of periodic function into Fourier series - Fourier series for odd and even functions - Hall-range Fourier cosine and sine series - Change of interval - Related problems.

UNIT IV FOURIER TRANSFORMS

(12 Hrs)

Fourier Integral theorem. Fourier transform and its inverse, properties. Fourier sine and cosine transforms, their properties, Convolution and Parseval's identity.

UNIT V Z-TRANSFORMS

(12 Hrs)

Difference equations, basic definition, z-transform - definition, Standard z-transforms, Damping rule, Shifting rule, Initial value and final value theorems and problems, Inverse z-transform. Applications of z-transforms to solve difference equations.

            """,
            "Data Structure and Applications": """
UNIT I BASIC TERMINOLOGIES OF DATA STRUCTURES

Introduction: Basic Terminologies - Elementary Data Organizations. Data Structure Operations: Insertion Deletion-Traversal. Array and its operations. Polynomial Manipulation

UNIT II STACK AND QUEUE OPERATIONS

(9 Hrs)

Stacks and Queues: ADT Stack and its operations. Applications of Stacks: Expression Conversion and evaluation. ADT Queue: Types of Queue - Simple Queue - Circular Queue - Priority Queue - Operations on each type of Queues.

UNIT III LINKED LIST OPERATIONS

(9 Hrs)

Linked Lists: Singly linked lists Representation in memory. Algorithms of several operations: Traversing Searching Insertion - Deletion in linked list. Linked representation of Stack and Queue. Doubly linked list: operations. Circular Linked Lists: operations.
UNIT IV TREES

(9 Hrs)

Trees: Basic Tree Terminologies Different types of Trees: Binary Tree Threaded Binary Tree - Binary Search Tree-Binary Tree Traversals - AVL Tree. Introduction to B-Tree and B+ Tree. Heap-Applications of heap.

(9 Hrs)

UNIT V HASHING AND GRAPHS

Hashing. Hash Table - Hash Function and its characteristics. Graph: Basic Terminologies and Representations - Graph traversal algorithms. Definition - Representation of Graph-Types of graph-Breadth-first traversal- Depth-first traversal-Topological Sort-Bi-connectivity-Cut vertex-Euler circuits-Applications of graphs.


            """,
            "Object Oriented Programming": """
                UNIT I OBJECT ORIENTED PROGRAMMING IN C++

Object Oriented Programming Concepts: Basic Program Construction Data Types Type Conversion - Operators-Key Concepts of Object Oriented Programming. Introduction and Structure of the C++ program- Stream Classes Formatted and Unformatted Data Unformatted Console I/O Operations Bit Fields Manipulators. Decision making statements-jump statement-switch case statement-looping statements

UNIT II CLASSES AND OBJECTS, CONSTRUCTORS AND DESTRUCTORS

Introduction to Classes and Objects Constructors and its Types Overloading Constructors Constructors-Destructors.

(9 Hrs)

Copy

UNIT III FUNCTIONS AND INHERITANCE

Functions: Passing arguments LValues and RValues. Library Functions Inline functions - Friend Functions. Inheritance: Introduction-Types of Inheritance.

(9 Hrs)

UNIT IV POLYMORPHISM AND VIRTUAL FUNCTION

(9 Hrs)

Polymorphism: Compile Time and Run Time Polymorphism. Overloading: Function Overloading and Operator Overloading Overloading Unary Operators - Overloading Binary Operators. Virtual Functions - Abstract Classes.

UNIT V TEMPLATES AND EXCEPTION HANDLING

(9 Hrs)

Generic Functions-Need of Templates - Function Templates-Class Templates. Exception Handling: Need of Exceptions - Keywords - Simple and Multiple Exceptions.

            """,
            "Computer and Communication Networks": """
UNIT I INTRODUCTION
Network Applications Network Hardware and Software OSI TCP/IP model - Example Networks - Internet protocols and standards Connection Oriented Network X25 Frame Relay Guided Transmission Media Wireless Transmission Mobile Telephone System Topologies. Case Study: Simple network communication with corresponding cables Transmission modes

UNIT II DATA LINK LAYER

(9 Hrs)

Framing-Error Detection and Correction-Checksum. DLC services-Sliding window protocols - Flow and HDLCPPP Multiple access protocols Multiplexing Ethernet- IEEE 802.11 Error control IEEE802.16-Bluetooth-RFID

UNIT III NETWORK LAYER

(9 Hrs)

Network layer services Packet Switching - IPV4 Addresses subnetting Routing algorithms. Network layer protocols: RIP-OSPF-BGP-ARP-DHCP-ICMP-IPv4 and IPv6-Mobile IP-Congestion control algorithms-Virtual Networks and Tunnels-Global Internet. Case study-Different routing algorithms to select the network path with its optimum and economical during data transfer Link State routing - Flooding - Distance vector.

UNIT IV TRANSPORT LAYER

(9 Hrs)

Introduction-Transport layer protocol - UDP-Reliable byte stream (TCP)-Connection management-Flow control-Retransmission-TCP Congestion control-Congestion avoidance-Queuing-QoS-Application requirements.

UNIT V APPLICATION LAYER

(9 Hrs) DNS-E-Mail-WWW-Architectural Overview - Dynamic web document and http. Protocols: SSH-SNMP -FTP-SMTP-SONET/SDH-ATM-Telnet-POP.

            """,
            "Computer Programming": """
UNIT I INTRODUCTION TO PYTHON

(9 Hrs)

Structure of Python Program Underlying mechanism of Module Execution Branching and Looping Problem Solving Using Branches and Loops - Functions - Lambda Functions-Lists and Mutability-Problem Solving Using Lists and Functions.

UNIT II SEQUENCE DATATYPES AND OBJECT ORIENTED PROGRAMMING

Sequences Mapping and Sets Dictionaries. Classes: Classes and Instances - Inheritance - Exception Handling-Introduction to Regular Expressions using "re" module.

(9 Hrs)

UNIT III USING NUMPY

(9 Hrs)

Basics of NumPy Computation on NumPy-Aggregations-Computation on Arrays - Comparisons - Masks and Boolean Arrays-Fancy Indexing - Sorting Arrays - Structured Data: NumPy's Structured Array.

(9 Hrs)

UNIT IV DATA MANIPULATION WITH PANDAS

Introduction to Pandas Objects Data indexing and Selection Operating on Data in Pandas Handling Missing Data Hierarchical Indexing Combining Data Sets. Aggregation and Grouping - Pivot Tables - Vectorized String Operations-Working with Time Series - High Performance Pandas - eval() and query().

(9 Hrs)

UNIT V VISUALIZATION WITH MATPLOTLIB

Basic functions of Matplotlib Simple Line Plot Scatter Plot - Density and Contour Plots - Histograms Binnings and Density-Customizing Plot Legends - Colour Bars-Three-Dimensional Plotting in Matplotlib.

            """
        }


        # Generate To-Do lists for all courses
        if st.button("Generate and Assign To-Do Lists"):
            with st.spinner("Generating To-Do Lists..."):
                todo_list = generate_todo_list_all_courses(courses, syllabus_dict)

                if todo_list:
                    st.session_state["todo_list"] = todo_list
                    st.success("To-Do Lists generated successfully!")

                    # Assign To-Do lists to students
                    def assign_todo(course, performance):
                        category_mapping = {
                            "Poor": "Weak",
                            "Medium": "Medium",
                            "High": "Strong"
                        }
                        normalized_performance = category_mapping.get(performance, performance)
                        return todo_list.get(course, {}).get(normalized_performance, [])

                    try:
                        data['To-Do List'] = data.apply(
                            lambda x: assign_todo(x['Course Name'], x['Performance Category']),
                            axis=1
                        )

                        st.subheader("Updated Data with To-Do Lists")
                        st.dataframe(data[['Course Name', 'Candidate Name', 'Performance Category', 'To-Do List']])
                        st.session_state["data_with_todo"] = data
                        # Save updated data to CSV
                        data.to_csv('anar3_final.csv', index=False)
                        st.success("Data with To-Do Lists saved to 'anar3_final.csv'.")
                    except Exception as e:
                        st.error(f"Error mapping To-Do lists to students: {e}")
                else:
                    st.error("Failed to generate To-Do lists. Please check the inputs.")
    else:
        st.warning("Please upload and process data in Anar1 first.")

elif selected_phase == "Anar4":
    st.title("Phase 4: Anar4")
    st.subheader("Find the poor performers in class. Pair them with good performers for Knowledge sharing & performance improvement.")
    if st.session_state["processed_data"] is not None:
        df = st.session_state["processed_data"]
        # Filter the dataset to keep only the highest attempt per candidate per course
        filtered_data_coursewise = df.loc[df.groupby(['Candidate Email', 'Course ID'])['Attempt ID'].idxmax()]

        # Sort the data for better clarity (optional)
        df_last_attempt = filtered_data_coursewise.sort_values(by=['Candidate Email', 'Course ID', 'Attempt ID']).reset_index(drop=True)
        # Identify poor performers and their strong performer pairs
        poor_performers = df_last_attempt[df_last_attempt['Performance Category'] == 'Poor']
        strong_performers_poor = df_last_attempt[df_last_attempt['Performance Category'] == 'High']
        unique_pairs_poor = set()
        pairs_poor = []
        # Pair poor performers with strong performers
        for _, poor in poor_performers.iterrows():
            # Find strong performers for the same course
            strong_candidates = strong_performers_poor[strong_performers_poor['Course Name'] == poor['Course Name']]
            
            for _, strong in strong_candidates.iterrows():
                pair_key = (poor['Candidate Email'], strong['Candidate Email'], poor['Course Name'])
                if pair_key not in unique_pairs_poor:
                    unique_pairs_poor.add(pair_key)
                    pairs_poor.append({
                        'Course Name': poor['Course Name'],
                        'Poor Performer': f"{poor['Candidate Name']} ({poor['Candidate Email']})",
                        'Strong Performer': f"{strong['Candidate Name']} ({strong['Candidate Email']})"
                    })
                    break  # Pair each poor performer only once
        st.subheader("Overall Poor Performer:")
        st.dataframe(poor_performers[["Candidate Name", "Candidate Email", "Course Name",'Performance Category']])
        pairs_poor_df = pd.DataFrame(pairs_poor)
        st.session_state["knowledge_sharing_pairs"] = pairs_poor_df
        pairs_poor_df.to_csv('knowledge_sharing_pairs_poor.csv', index=False)
        st.subheader("Knowledge Sharing Pairs For Poor:")
        st.dataframe(pairs_poor_df)

        # Identify medium performers and their strong performer pairs
        medium_performers = df_last_attempt[df_last_attempt['Performance Category'] == 'Medium']
        strong_performers_medium = df_last_attempt[df_last_attempt['Performance Category'] == 'High']

        unique_pairs_medium = set()
        pairs_medium = []

        # Pair medium performers with strong performers
        for _, medium in medium_performers.iterrows():
            # Find strong performers for the same course
            strong_candidates = strong_performers_medium[strong_performers_medium['Course Name'] == medium['Course Name']]
            
            for _, strong in strong_candidates.iterrows():
                pair_key = (medium['Candidate Email'], strong['Candidate Email'], medium['Course Name'])
                if pair_key not in unique_pairs_medium:
                    unique_pairs_medium.add(pair_key)
                    pairs_medium.append({
                        'Course Name': medium['Course Name'],
                        'Medium Performer': f"{medium['Candidate Name']} ({medium['Candidate Email']})",
                        'Strong Performer': f"{strong['Candidate Name']} ({strong['Candidate Email']})"
                    })
                    break

        pairs_medium_df = pd.DataFrame(pairs_medium)
        st.session_state["knowledge_sharing_pairs_mediums"] = pairs_medium_df
        pairs_medium_df.to_csv('knowledge_sharing_pairs_medium.csv', index=False)
        st.subheader("Knowledge Sharing Pairs For Medium:")
        st.dataframe(pairs_medium_df)

    else:
        st.warning("Please upload and process data in Anar1 first.")

elif selected_phase == "Anar5":
    st.title("Phase 5: Anar5")
    st.subheader("Highlight data on who is improving in the class based on the ToDo-Course completion.")
    # Load current data from session state
    if st.session_state["processed_data"] is not None:
        current_file = "./final.csv"
        current_data = pd.read_csv(current_file)
        # Filter the dataset to keep only the highest attempt per candidate per course
        filtered_data_coursewise = current_data.loc[current_data.groupby(['candidate_email', 'course_id'])['attempt'].idxmax()]
        # Sort the data for better clarity (optional)
        current_data = filtered_data_coursewise.sort_values(by=['candidate_email', 'course_id', 'attempt']).reset_index(drop=True)
        # Load future data from the given path
        future_file_path = "./updated_student_marks.csv"
        try:
            future_data = pd.read_csv(future_file_path)
            # Filter the dataset to keep only the highest attempt per candidate per course
            filtered_data_coursewise = future_data.loc[future_data.groupby(['candidate_email', 'course_id'])['attempt'].idxmax()]
            # Sort the data for better clarity (optional)
            future_data = filtered_data_coursewise.sort_values(by=['candidate_email', 'course_id', 'attempt']).reset_index(drop=True)
        except FileNotFoundError:
            st.error(f"Future data file not found at: {future_file_path}. Please check the file path.")
            future_data = None
                
        # Perform comparison if both current and future data are available
        if "processed_data" in st.session_state and future_data is not None:
            # Comparison Logic
            def compare_performance(current_df, future_df):
                # Merge the DataFrames on `candidate_name` and `course_id`
                merged_df = pd.merge(
                    current_df,
                    future_df,
                    on=["candidate_name", "course_id"],
                    suffixes=("_current", "_future"),
                    how="inner"
                )

                # Dictionaries to store subject-wise data
                improved_dict = {}
                not_improved_dict = {}

                for _, row in merged_df.iterrows():
                    candidate = row["candidate_name"]
                    course = row["course_name_future"]

                    # Compare marks
                    mark_current = row["mark_current"]
                    mark_future = row["mark_future"]
                    mark_improved = mark_future > mark_current

                    # Compare grades
                    grade_current = row["grade_current"]
                    grade_future = row["grade_future"]
                    grade_reduced = grade_future < grade_current  # Grade reduced for not improved

                    # Compare Performance Category
                    perf_current = row["Performance_category_current"]
                    perf_future = row["Performance_category_future"]

                    # Conditions for performance improvement
                    perf_improved = False
                    if perf_current == "Poor" and perf_future == "Medium":
                        perf_improved = True
                    elif perf_current == "Medium" and perf_future == "High":
                        perf_improved = True
                    elif perf_current == "High" and perf_future == "High":
                        perf_improved = True

                    # Conditions for performance worsening
                    perf_worsened = False
                    if perf_current == "Poor" and perf_future == "Poor":
                        perf_worsened = True
                    elif perf_current == "Medium" and perf_future == "Poor":
                        perf_worsened = True
                    elif perf_current == "High" and perf_future == "Medium":
                        perf_worsened = True

                    # Create student data
                    student_data = {
                        "Candidate Name": candidate,
                        "Mark Comparison": f"{mark_current} -> {mark_future}",
                        "Grade Comparison": f"{grade_current} -> {grade_future}",
                        "Performance Comparison": f"{perf_current} -> {perf_future}",
                    }

                    # Categorize into improved or not improved, grouped by subject
                    if mark_improved or perf_improved:
                        if course not in improved_dict:
                            improved_dict[course] = []
                        improved_dict[course].append(student_data)
                    elif grade_reduced or perf_worsened:
                        if course not in not_improved_dict:
                            not_improved_dict[course] = []
                        not_improved_dict[course].append(student_data)

                # Convert dictionaries to DataFrames
                improved_subject_dfs = {
                    course: pd.DataFrame(data) for course, data in improved_dict.items()
                }
                not_improved_subject_dfs = {
                    course: pd.DataFrame(data) for course, data in not_improved_dict.items()
                }

                # Count unique names per subject
                improved_counts = {
                    course: len(df["Candidate Name"].unique())
                    for course, df in improved_subject_dfs.items()
                }
                not_improved_counts = {
                    course: len(df["Candidate Name"].unique())
                    for course, df in not_improved_subject_dfs.items()
                }

                return improved_subject_dfs, not_improved_subject_dfs, improved_counts, not_improved_counts

            # Run the Comparison
            improved_subject_wise, not_improved_subject_wise, improved_count, not_improved_count = compare_performance(
                current_data, future_data
            )

            # Store results in session state
            st.session_state["improved_students"] = improved_subject_wise
            st.session_state["not_improved_students"] = not_improved_subject_wise
            st.session_state["improved_count"] = improved_count
            st.session_state["not_improved_count"] = not_improved_count

            # Visualize the counts
            st.write("### Subject-Wise Improvement Overview")
            # Prepare data for the table
            improvement_data = {
                "Course Name": list(improved_count.keys()),
                "Improved Count": list(improved_count.values()),
            }
            # Convert to a DataFrame
            improvement_df = pd.DataFrame(improvement_data)
            # Display the table
            st.write("#### Improved Students by Course")
            st.dataframe(improvement_df, use_container_width=True)

            # Display Detailed Tables
            st.write("### Improved Students (Subject-Wise):")
            if st.session_state["improved_students"]:
                for subject, df in st.session_state["improved_students"].items():
                    st.write(f"#### {subject}")
                    st.dataframe(df)
            else:
                st.write("No students showed improvement in any subject.")
    else:
        st.info("Please ensure anar1 are loaded to proceed.")
    
elif selected_phase == "Anar6":
    st.title("Phase 6: Anar6")
    st.subheader("Provide a Dashboard to get an overall class performance.")
    # Add custom CSS for styling
    st.markdown("""
        <style>
        .stMetric-value {
            color: #2E86C1;
            font-weight: bold;
        }
        .main-title {
            text-align: center;
            color: #2ECC71;  /* Green title */
        }
        .section-title {
            color: #5D6D7E;  /* Gray section headers */
        }
        .center-options {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.markdown('<h1 class="main-title">📊 Course Performance Dashboard</h1>', unsafe_allow_html=True)

    if st.session_state["processed_data"] is not None:
        # Load CSV into a DataFrame
        df = pd.read_csv('./final.csv')
       # Filters
        selected_course = st.selectbox("📚 Select a Course", ["All"] + list(df["course_name"].unique()))
        selected_category = st.selectbox("🏷 Select Performance Category", ["All"] + list(df["Performance_category"].unique()))
        min_marks, max_marks = st.slider(
            "🎯 Select Marks Range",
            min_value=int(df["mark"].min()),
            max_value=int(df["mark"].max()),
            value=(int(df["mark"].min()), int(df["mark"].max()))
        )

        # Apply filters
        filtered_df = df
        if selected_course != "All":
            filtered_df = filtered_df[filtered_df["course_name"] == selected_course]
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["Performance_category"] == selected_category]
        filtered_df = filtered_df[(filtered_df["mark"] >= min_marks) & (filtered_df["mark"] <= max_marks)]

        # Display key insights
        st.markdown('<h3 class="section-title">📊 Key Insights</h3>', unsafe_allow_html=True)
        total_candidates = filtered_df["candidate_name"].nunique()
        avg_marks = filtered_df["mark"].mean()

        # Calculate top performance category based on filtered data
        if not filtered_df.empty:
            top_performance = (
                filtered_df["Performance_category"].value_counts().idxmax()
            )
        else:
            top_performance = "No Data"

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", total_candidates, "👤")
        col2.metric("Average Marks", f"{avg_marks:.2f}" if total_candidates > 0 else "0.00", "📈")
        col3.metric("Top Performance Category", top_performance, "🏆")

        # Display filtered data
        st.markdown('<h3 class="section-title">🔍 Filtered Data</h3>', unsafe_allow_html=True)
        st.dataframe(filtered_df, use_container_width=True)

        # Visualization options in the center
        st.markdown('<h3 class="section-title">📊 Visualization Options</h3>', unsafe_allow_html=True)
        visualization_type = st.radio(
            "Choose a Visualization",
            [
                "Performance Category Distribution", "Marks Distribution", "Line Chart by Performance",
                "Marks vs Attempts", "Top Courses by Average Marks", "Heatmap Correlation"
            ],
            horizontal=True
        )

        # Visualizations
        if visualization_type == "Performance Category Distribution":
            st.subheader("📈 Performance Category Distribution")
            category_counts = filtered_df["Performance_category"].value_counts()
            fig = px.pie(
                names=category_counts.index,
                values=category_counts.values,
                title="Performance Category Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Marks Distribution":
            st.subheader("📊 Marks Distribution")
            bins = st.slider("🔧 Adjust Number of Bins", min_value=5, max_value=50, value=20)
            fig = px.histogram(
                filtered_df, x="mark", nbins=bins, title="Marks Distribution",
                labels={"mark": "Marks"}, marginal="box", color_discrete_sequence=["#F39C12"]
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Line Chart by Performance":
            st.subheader("📈 Line Chart by Performance")
            # Aggregate the data for the line chart
            performance_line_data = filtered_df.groupby("Performance_category")["mark"].mean().reset_index()
            
            # Create the line chart
            fig = px.line(
                performance_line_data,
                x="Performance_category",
                y="mark",
                title="Line Chart of Average Marks by Performance Category",
                labels={"Performance_category": "Performance Category", "mark": "Average Marks"},
                markers=True,  # Add markers to the line
            )
            st.plotly_chart(fig, use_container_width=True)


        elif visualization_type == "Marks vs Attempts":
            st.subheader("📌 Marks vs Attempts")
            fig = px.scatter(
                filtered_df, x="attempt", y="mark", color="Performance_category",
                title="Marks vs Attempts",
                labels={"attempt": "Attempts", "mark": "Marks"},
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Top Courses by Average Marks":
            st.subheader("🏅 Top Courses by Average Marks")
            top_courses = filtered_df.groupby("course_name")["mark"].mean().sort_values(ascending=False).head(10)
            fig = px.bar(
                x=top_courses.values, y=top_courses.index,
                orientation="h",
                labels={"x": "Average Marks", "y": "Course Name"},
                title="Top 10 Courses by Average Marks",
                color=top_courses.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            st.plotly_chart(fig, use_container_width=True)

        elif visualization_type == "Heatmap Correlation":
            st.subheader("🔥 Heatmap of Correlation")
            corr = filtered_df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            st.pyplot(fig)
    else:
        st.info("📥 Please upload a CSV file to get started.")