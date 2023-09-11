import numpy as np
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

pd.set_option('display.max_columns', None)

def confusion_matrix_plot(model, y_test, y_predictions):
    plt.rcParams['figure.figsize']=12,12
    sns.set_style("white")
    cm = confusion_matrix(y_test, y_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation = 270)
    plt.show()

# if we want a feature importance plot, we can run a model and call this function
def feature_importance_plot(x, model):
    feature_names = list(x.columns)
    plt.rcParams['figure.figsize']=7,7
    sns.set_style("white")
    feat_importances = pd.Series(model.feature_importances_, index=feature_names)
    feat_importances = feat_importances.groupby(level=0).mean()
    feat_importances.nlargest(20).plot(kind='barh').invert_yaxis()
    sns.despine()
    plt.show()


# if we want to visualized the results of a particular model, the scores can be passed in and printed
def print_results(accuracy, recall, precision, f1s):
    print("Accuracy: "+ "{:.2%}".format(accuracy))
    print("Recall: "+ "{:.2%}".format(recall))
    print("Precision: "+ "{:.2%}".format(precision))
    print("F1-Score: "+ "{:.2%}".format(f1s))


# this function 
def record_output(model_performance, y_test, y_predictions, algorithm_name, name):
    accuracy = accuracy_score(y_test, y_predictions)
    recall = recall_score(y_test, y_predictions, average='weighted')
    precision = precision_score(y_test, y_predictions, average='weighted')
    f1s = f1_score(y_test, y_predictions, average='weighted')

    row_index = len(model_performance)
    model_performance.loc[row_index] = [algorithm_name, accuracy, recall, precision, f1s, name]
    return model_performance


def logistical_classification(X_train, X_test, y_train):
        model = LogisticRegression(random_state=42, max_iter=3000).fit(X_train,y_train)
        y_predictions = model.predict(X_test) # These are the predictions from the test data
        return y_predictions


def K_neighbors(X_train, X_test, y_train):
        # does not take random_state arg
        model = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)
        y_predictions = model.predict(X_test) # These are the predictions from the test data.
        return y_predictions


def decision_tree(X_train, X_test, y_train):
        model = DecisionTreeClassifier(random_state=42).fit(X_train,y_train)
        y_predictions = model.predict(X_test) # These are the predictions from the test data
        return y_predictions


def extra_trees(X_train, X_test, y_train):
        model = ExtraTreesClassifier(random_state=42,n_jobs=-1).fit(X_train,y_train)
        y_predictions = model.predict(X_test) # These are the predictions from the test data
        return y_predictions


def random_forest(X_train, X_test, y_train):
    model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,bootstrap=True, random_state = 42).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    return model, y_predictions


def gradient_boost(X_train, X_test, y_train):
    model = GradientBoostingClassifier(random_state=42).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    return y_predictions


def neural_network(X_train, X_test, y_train):
    model = MLPClassifier(hidden_layer_sizes = (20,20,), 
                                  activation='relu', 
                                  solver='adam',
                                  batch_size=2000,
                                  verbose=0,
                                  random_state=42).fit(X_train,y_train)
    y_predictions = model.predict(X_test) # These are the predictions from the test data.
    return y_predictions


def recode_class(y_test, y_predicitions):
    # passes in y_test and y_predictions from the ML run
    y_test_recoded = pd.Series(y_test).where(lambda x: x.str.contains('Normal')==False, 'Normal')
    y_predictions_recoded = pd.Series(y_predicitions).where(lambda x: x.str.contains('Normal')==False, 'Normal')
    return y_test_recoded, y_predictions_recoded


def recode_label(y_test, y_predicitions):
    # passes in y_test and y_predictions from the ML run
    y_test_recoded = y_test.apply(lambda x: 0 if 'Normal' in x else 1)
    y_predictions_recoded = pd.Series(y_predicitions).apply(lambda x: 0 if 'Normal' in x else 1)
    return y_test_recoded, y_predictions_recoded

# breaks up traffic into normal and attack
def separate_traffic_cats(df):
    normals = df.drop(df[df['label'] == 1].index).reset_index(drop=True)
    attacks = df.drop(df[df['label'] == 0].index).reset_index(drop=True)
    return normals, attacks


# one hot encodes and makes sure the cols are the correct categories
def preprocess(df, features):
    df['is_ftp_login'] = np.where(df['is_ftp_login'] > 1, 1, df['is_ftp_login'])
    df['service'].replace('-', np.nan, inplace=True)
    features['Type '] = features['Type '].str.lower()
    features['Name'] = features['Name'].str.lower()
    integer_names = features['Name'][features['Type '] == 'integer']
    binary_names = features['Name'][features['Type '] == 'binary']
    float_names = features['Name'][features['Type '] == 'float']
    cols = df.columns
    integer_names = cols.intersection(integer_names)
    binary_names = cols.intersection(binary_names)
    float_names = cols.intersection(float_names)

    for c in integer_names:
        pd.to_numeric(df[c])
    for c in binary_names:
        pd.to_numeric(df[c])
    for c in float_names:
        pd.to_numeric(df[c])

    num_col = df.select_dtypes(include='number').columns

    # selecting categorical data attributes
    cat_col = df.columns.difference(num_col)

    data_cat = df[cat_col].copy()
    data_cat = pd.get_dummies(data_cat, columns=cat_col)
    df = pd.concat([df, data_cat], axis=1)
    df.drop(columns=cat_col, inplace=True)
    df.replace([False, True], [0, 1], inplace = True) # look this over one more time
    return df

# removes the ground truth from the df
def remove_ground_truth(df, no_copy=True):  # set no_copy to False to copy the make a copy, True for in-place change
    ground_truth = df[['attack_cat', 'label']]
    df.drop(['attack_cat'], axis=1, inplace=no_copy)
    df.drop(['label'], axis=1, inplace=no_copy)
    return df, ground_truth


# a function that maps the vector from hdbscan onto the df to create the benign sub categories
def hdbscan_df_transformer(pickle_file, df):
    model = pickle.load(open(pickle_file, 'rb'))
    # extracting the sample and cluster sizes from the pickle file name
    file_list = list(filter(lambda x: x.isdigit(), pickle_file.split('_')))
    params = [int(s) for s in file_list]
    scan = pd.DataFrame(model, columns = [f'attack_cat_bc_{params[0]}_{params[1]}'])
    # print(scan.shape)
    # print(scan.head(500))
    # create the clustered columns with the parameters taken from the pickle file name
    df_concat = pd.concat([df, scan], axis = 1).reset_index(drop=True)
    # print('df_concat shape before concat: ' + str(df_concat.shape))
    df_concat[f'attack_cat_bc_{params[0]}_{params[1]}'] = 'Normal_' + df_concat[f'attack_cat_bc_{params[0]}_{params[1]}'].astype(str)
    df_concat[f'attack_cat_bc_{params[0]}_{params[1]}'].astype('category')
    # print('df_concat shape after concat: ' + str(df_concat.shape))
    return df_concat


# normalizes the passed in df
def normalization(df, col):
    minmax_scale = MinMaxScaler(feature_range=(0, 1))
    for i in col:
        if i == 'id':  # don't transform the ID column
            continue
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
    return df


# splits the df into train and test and creates a column with the appropriate labels
def train_test_splitter(df):
    # make x the entire df
    x = df
    # the y can be anything since it doesn't get used to make the 'train_test' column
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify=y)
    # once I have x_train and x_test, create a column in each called train_test and populate that column with either 0s or 1s
    x_train['train_test'] = 0
    x_test['train_test'] = 1
    # put train and test together
    df_concat = pd.concat([x_train, x_test], axis = 0).reset_index(drop=True)
    return df_concat

# run_it_all splits up the data into train and test variables based on the 'train_test' column and the two attack_cat columns
# (without benign clusters and with), runs then through 7 ML Classifiers and adds output into Model Performance, which gets returned at the end
# anomaly detection is built in to occur at each stage: base, classified, and rejoined
def run_it_all(data):
    # break the data into train_test and then x and y_base, y_class
    x_train = data["train"].loc[:, :'state_RST'] # get everything before and including 'state_RST' because the ground truth cols are after that
    x_test = data["test"].loc[:, :'state_RST']
    y_train_base = data["train"]['attack_cat']; y_train_class = data["train"].iloc[:, -2]
    # will take out the column before train_test regardless of it's name, but I couldn't figure out an effective way to extract it's name that could vary
    y_test_base = data["test"]['attack_cat']; y_test_class = data["test"].iloc[:, -2]
    # print(y_train_class.head())

    # the ML classifiers that will be used and their matching funciions:
    algorithms = [('Logistical Classification', logistical_classification), ('K Neighbors', K_neighbors), ('Decision Tree', decision_tree), 
        ('Extra Trees', extra_trees),('Random Forest', random_forest),('Gradient Boost', gradient_boost),('Neural Network', neural_network)]

    # create the model_Performance df
    model_performance = pd.DataFrame(columns=['Algorithm', 'Accuracy','Recall','Precision','F1-Score', 'Name'])

    num_rows = str(len(data.index))[:-3] + 'k'

    # run the algorithms
    for algorithm_name, algorithm_func in tqdm(algorithms):
        # base:
        y_predictions = algorithm_func(x_train, x_test, y_train_base)
        model_performance = record_output(model_performance, y_test_base, y_predictions, algorithm_name, (num_rows + ' Base'))
        y_test_base_anom, y_predictions_base_anom = recode_label(y_test_base, y_predictions)
        model_performance = record_output(model_performance, y_test_base_anom, y_predictions_base_anom, algorithm_name, (num_rows + ' Base AD'))

        # classified:
        y_predictions = algorithm_func(x_train, x_test, y_train_class)
        model_performance = record_output(model_performance, y_test_class, y_predictions, algorithm_name, (num_rows + ' Classified'))
        y_test_class_anom, y_predictions_class_anom = recode_label(y_test_class, y_predictions)
        model_performance = record_output(model_performance, y_test_class_anom, y_predictions_class_anom, algorithm_name, (num_rows + ' Classified AD'))

        # rejoined:
        y_test_recoded, y_predictions_recoded = recode_class(y_test_class, y_predictions)
        model_performance = record_output(model_performance, y_test_recoded, y_predictions_recoded, algorithm_name, (num_rows + ' Rejoined'))
        y_test_rejoined_anom, y_predictions_rejoined_anom = recode_label(y_test_recoded, y_predictions_recoded)
        model_performance = record_output(model_performance, y_test_rejoined_anom, y_predictions_rejoined_anom, algorithm_name, (num_rows + ' Rejoined AD'))

    return model_performance
