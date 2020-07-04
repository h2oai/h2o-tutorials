"""H2O-3 RuleFit"""

# Contributors: Megan Kurka - megan.kurka@h2o.ai
# Created: February 18th, 2020
# Last Updated: April 30th, 2020

import pandas as pd
import numpy as np
import os
import warnings

import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator
from h2o.exceptions import H2OValueError
from h2o.tree import H2OTree

class H2ORuleFit():
    """
    H2O RuleFit
    Builds a Distributed RuleFit model on a parsed dataset, for regression or 
    classification. 
    :param algorithm: The algorithm to use to generate rules.  Options are "DRF", "XGBoost", "GBM"
    :param min_rule_len: Minimum length of rules. Defaults to 1.
    :param max_rule_len: Maximum length of rules. Defaults to 10.
    :param max_num_rules: The maximum number of rules to return.
            Defaults to None which means the number of rules is selected by diminishing returns in model deviance.
    :param nfolds: Number of folds for K-fold cross-validation. Defaults to 5.
    :param seed: Seed for pseudo random number generator. Defaults to -1.
    :param tree_params: Additional parameters that can be passed to the tree model. Defaults to None.
    :param glm_params: Additional parameters that can be passed to the linear model. Defaults to None.
    :returns: a set of rules and coefficients
    :examples:
    >>> rulefit = H2ORuleFit()
    >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
    ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
    >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
    >>> rulefit.train(x=x,y="survived",training_frame=training_data)
    >>> rulefit
    """
    
    def __init__(self, algorithm,
                 min_rule_len=1, 
                 max_rule_len=10, 
                 max_num_rules=None, 
                 nfolds=5, 
                 seed=-1,
                 tree_params={},
                 glm_params={}
                ):
        
        if algorithm not in ["DRF", "XGBoost", "GBM"]:
            raise H2OValueError("{} is not a supported algorithm".format(algorithm))
        self.algorithm = algorithm
        self.min_rule_len = min_rule_len
        self.max_rule_len = max_rule_len
        self.max_num_rules = max_num_rules
        self.nfolds = nfolds
        self.seed = seed
        
        if tree_params:
            tree_params.pop("model_id", None)
            if 'max_depth' in tree_params.keys():
                self.min_rule_len = tree_params.get("max_depth")
                self.max_rule_len = tree_params.get("max_depth")
                tree_params.pop("max_depth")
                warnings.warn('max_depth provided in tree_params - min_rule_len and max_rule_len will be ignored')
            if 'nfolds' in tree_params.keys():
                tree_params.pop('nfolds')
                warnings.warn('seed provided in tree_params but will be ignored')
            if 'seed' in tree_params.keys():
                tree_params.pop('seed')
                warnings.warn('seed provided in tree_params but will be ignored')
            
                
        if glm_params:
            glm_params.pop("model_id", None)
            if 'max_active_predictors' in glm_params.keys():
                self.max_num_rules = glm_params.get("max_active_predictors") - 1
                glm_params.pop("max_active_predictors")
                warnings.warn('max_active_predictors provided in glm_params - max_num_rules will be ignored')
            if 'nfolds' in glm_params.keys():
                glm_params.pop('nfolds')
                warnings.warn('seed provided in glm_params but will be ignored')
            if 'seed' in glm_params.keys():
                glm_params.pop('seed')
                warnings.warn('seed provided in glm_params but will be ignored')
            if 'alpha' in glm_params.keys():
                glm_params.pop('alpha')
                warnings.warn('alpha ignored - set to 1 by rulefit')
            if 'lambda_' in glm_params.keys():
                glm_params.pop('lambda_')
                warnings.warn('lambda_ ignored by rulefit')
                
        self.tree_params = tree_params
        self.glm_params = glm_params
            
        
    def train(self, x=None, y=None, training_frame=None):
        """
        Train the rulefit model.
        :param x: A list of column names or indices indicating the predictor columns.
        :param y: An index or a column name indicating the response column.
        :param training_frame: The H2OFrame having the columns indicated by x and y (as well as any
            additional columns specified by fold, offset, and weights).
        :examples:
        >>> rulefit = H2ORuleFit()
        >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
        ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
        >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
        >>> rulefit.train(x=x,y="survived",training_frame=training_data)
        >>> rulefit
        """
        
        if (training_frame.type(y) == "enum"):
            if training_frame[y].unique().nrow > 2:
                family = "multinomial"
                raise H2OValueError("multinomial use cases not yet supported")
            else:
                family = "binomial"
        else:
            if self.glm_params.get("family") is not None:
                family = self.glm_params.get("family")
                self.glm_params.pop("family")
            else:
                family = "gaussian"


        # Get paths from random forest models
        paths_frame = training_frame[y]
        depths = range(self.min_rule_len, self.max_rule_len + 1)
        tree_models = dict()
        for model_idx in range(len(depths)):

            # Train tree models
            tree_model = _tree_model(self.algorithm, 
                                     depths[model_idx], 
                                     self.seed,
                                     model_idx,
                                     self.tree_params
                                    )
            tree_model.train(y = y, x = x, training_frame = training_frame)
            tree_models[model_idx] = tree_model

            paths = tree_model.predict_leaf_node_assignment(training_frame)
            paths.col_names = ["tree_{0}.{1}".format(str(model_idx), x) for x in paths.col_names]
            paths_frame = paths_frame.cbind(paths)
            
        if self.max_num_rules:
            # Train GLM with chosen lambda
            glm = H2OGeneralizedLinearEstimator(model_id = "glm.hex", 
                                                seed = self.seed,
                                                family = family,
                                                alpha = 1, 
                                                max_active_predictors = self.max_num_rules + 1,
                                                **self.glm_params
                                               )
            glm.train(y = y, training_frame=paths_frame)
            
        else:
            # Get optimal lambda
            glm = H2OGeneralizedLinearEstimator(model_id = "glm.hex", 
                                                nfolds = self.nfolds, 
                                                seed = self.seed,
                                                family = family,
                                                alpha = 1, 
                                                lambda_search = True,
                                                **self.glm_params
                                               )
            glm.train(y = y, training_frame=paths_frame)
            
            lambda_ = _get_glm_lambda(glm)

            # Train GLM with chosen lambda
            glm = H2OGeneralizedLinearEstimator(model_id = "glm.hex", 
                                                seed = self.seed,
                                                family = family,
                                                alpha = 1, 
                                                lambda_ = lambda_,
                                                solver = "COORDINATE_DESCENT",
                                                **self.glm_params
                                               )
            glm.train(y = y, training_frame=paths_frame)
        
        # Get Intercept
        intercept = _get_intercept(glm)
        
        # Get Rules
        rule_importance = _get_rules(glm, tree_models, self.algorithm)
        
        self.intercept = intercept
        self.rule_importance = rule_importance
        self.glm = glm
        self.tree_models = tree_models
        
    def predict(self, test_data):
        """
        Predict on a dataset.

        :param H2OFrame test_data: Data on which to make predictions.

        :returns: A new H2OFrame of predictions.
        """
        paths_frame = test_data[0]
        for model_idx in self.tree_models.keys():

            paths = self.tree_models.get(model_idx).predict_leaf_node_assignment(test_data)
            paths.col_names = ["tree_{0}.{1}".format(str(model_idx), x) for x in paths.col_names]
            paths_frame = paths_frame.cbind(paths)
            
        paths_frame = paths_frame[1::]
        
        return self.glm.predict(paths_frame)
    
    def filter_by_rule(self, test_data, rule):
        """
        Returns records that match a provided rule.

        :param H2OFrame test_data: Data on which to find rule assignment.
        :param rule: The rule to use.

        :returns: A new H2OFrame of records that match the rule.
        """
        family = self.glm.params.get('family').get('actual')
        model_idx, tree_num, tree_class, path = _map_column_name(rule, family, self.algorithm)
        paths = self.tree_models.get(model_idx).predict_leaf_node_assignment(test_data)
        
        paths_col = ".".join(rule.split(".")[1:-1])
        paths_path = rule.split(".")[-1]
        
        return paths[paths[paths_col] == paths_path]
    
    def coverage_table(self, test_data):
        """
        Returns table of coverage per rule

        :param H2OFrame test_data: Data on which to find rule assignment.

        :returns: A new table with rule coefficients plus coverage
        """
        rules = self.rule_importance.copy(deep = True)
        coverage = [len(self.filter_by_rule(test_data, x)) for x in rules.variable.values]
        coverage_percent = [x/len(test_data) for x in coverage]
        
        rules["coverage_count"] = coverage
        rules["coverage_percent"] = coverage_percent
        
        return rules
    
    def varimp_plot(self, num_rules = 10):
        """
        Generate variable importanec plot of rules
        :param num_rules: The number of rule to graph.  Defaults to 10.
        :examples:
        >>> rulefit = H2ORuleFit()
        >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
        ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
        >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
        >>> rulefit.train(x=x,y="survived",training_frame=training_data)
        >>> rulefit.varimp_plot()
        """
        import plotly.graph_objects as go
        plot_data = self.rule_importance.copy(deep = True)
        if len(plot_data) > num_rules:
            plot_data = plot_data.iloc[0:num_rules]
        plot_data["color"] = np.where(plot_data.coefficient > 0, 'crimson', 'lightslategray')
        plot_data = plot_data.iloc[::-1]
        fig = go.Figure([go.Bar(x=plot_data.coefficient, y=plot_data.rule, marker_color = plot_data.color, orientation='h')])
        fig.update_layout(showlegend=False)
        return fig

    
    def save(self, path):
        """
        Save the rulefit model.
        :param path: The path to the directory where the models should be saved.
        :examples:
        >>> rulefit = H2ORuleFit()
        >>> training_data = h2o.import_file("smalldata/gbm_test/titanic.csv", 
        ...                                  col_types = {'pclass': "enum", 'survived': "enum"})
        >>> x = ["age", "sibsp", "parch", "fare", "sex", "pclass"]
        >>> rulefit.train(x=x,y="survived",training_frame=training_data)
        >>> rulefit.save(dir_path = "/home/user/my_rulefit/")
        """
        # save random forest models
        for tree_model in self.tree_models.values():
            h2o.save_model(tree_model, path=path)
            
        # save glm model
        h2o.save_model(self.glm, path=path)
        
        return path

    def load(self, path):
        """
        Load the saved rulefit model.
        :param path: The path to the rulefit model.
        :examples:
        >>> rulefit = H2ORuleFit()
        >>> rulefit.load(path)
        """
        # load GLM model
        glm = h2o.load_model(os.path.join(path, 'glm.hex'))
            
        # load tree models
        depths = range(self.min_rule_len, self.max_rule_len + 1)
        tree_models = dict()
        for model_idx in range(len(depths)):
            tree_models[model_idx] = h2o.load_model(os.path.join(path, "tree_{}.hex".format(model_idx)))
            
        # Get Intercept
        intercept = _get_intercept(glm)
        
        # Get Rules
        rule_importance = _get_rules(glm, tree_models, self.algorithm)
        
        self.intercept = intercept
        self.rule_importance = rule_importance
        self.glm = glm
        self.tree_models = tree_models


def _tree_model(algorithm, max_depth, seed, model_idx, tree_params):
    
    if algorithm == "DRF":
        # Train random forest models
        from h2o.estimators.random_forest import H2ORandomForestEstimator
        model = H2ORandomForestEstimator(seed = seed, 
                                         model_id = "tree_{}.hex".format(str(model_idx)), 
                                         max_depth = max_depth,
                                         **tree_params
                                        )
    elif algorithm == "GBM":
        
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        model = H2OGradientBoostingEstimator(seed = seed, 
                                             model_id = "tree_{}.hex".format(str(model_idx)), 
                                             max_depth = max_depth,
                                             **tree_params
                                            )
        
    elif algorithm == "XGBoost":
        from h2o.estimators.xgboost import H2OXGBoostEstimator
        model = H2OXGBoostEstimator(seed = seed, 
                                    model_id = "tree_{}.hex".format(str(model_idx)), 
                                    max_depth = max_depth,
                                    **tree_params
                                   )
        
    else:
        raise H2OValueError("{} algorithm not supported".format(algorithm))
        
    return model
    
        
def _get_glm_lambda(glm):
    """
    Get the best GLM lambda by choosing one diminishing returns on explained deviance
    """
    r = H2OGeneralizedLinearEstimator.getGLMRegularizationPath(glm)
    deviance = r.get('explained_deviance_train')
    if len(deviance) < 5:
        lambda_index = len(deviance) - 1
    else:
        lambda_index = [i*3 for i, x in enumerate(np.diff(np.sign(np.diff(deviance, 2)))) if x != 0 and i > 0][0]
        
    return r.get('lambdas')[lambda_index]

def _tree_traverser(node, split_path):
    """
    Traverse the tree to get the rules for a specific split_path
    """
    rule = []
    splits = [char for char in split_path]
    for i in splits:
        if i == "R":
            if np.isnan(node.threshold):
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.right_levels, 
                                'operator': 'in'}]
            else:
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.threshold, 
                                'operator': '>='}]

            node = node.right_child
        if i == "L":
            if np.isnan(node.threshold):
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.left_levels, 
                                'operator': 'in'}]

            else:
                rule = rule + [{'split_feature': node.split_feature, 
                                'value': node.threshold, 
                                'operator': '<'}]

            node = node.left_child
    consolidated_rules = _consolidate_rules(rule)
    consolidated_rules = " AND ".join(consolidated_rules.values())
    return consolidated_rules

def _consolidate_rules(rules):
    """
    Consolidate rules to remove redundancies
    """
    rules = [x for x in rules if x.get("value")]
    features = set([x.get('split_feature') for x in rules])
    consolidated_rules = {}
    for i in features:
        feature_rules = [x for x in rules if x.get('split_feature') == i]
        if feature_rules[0].get('operator') == 'in':
            cleaned_rules = i + " is in " + ", ".join(sum([x.get('value') for x in feature_rules], []))
        else:
            cleaned_rules = []
            operators = set([x.get('operator') for x in feature_rules])
            for op in operators:
                vals = [x.get('value') for x in feature_rules if x.get('operator') == op]
                if '>' in op:
                    constraint = max(vals)
                else:
                    constraint = min(vals)
                cleaned_rules = " and ".join([op + " " + str(round(constraint, 3))])
                cleaned_rules = i + " " + cleaned_rules
        consolidated_rules[i] = cleaned_rules
    
    return consolidated_rules

def _get_intercept(glm):
    """
    Get Intercept from GLM model
    """    
    family = glm.params.get('family').get('actual')
    # Get paths
    if family == "multinomial":
        intercept = {k: {k1: v1 for k1, v1 in v.items() if k1 == "Intercept"} for k, v in glm.coef().items()}
    else:
        intercept = {k: v for k, v in glm.coef().items() if k == 'Intercept'}
    return intercept
        
def _get_rules(glm, tree_models, algorithm):
    """
    Get Rules from GLM model
    """
    
    family = glm.params.get('family').get('actual')
    
    if family != "multinomial":
        coefs = {'coefs_class_0': glm.coef()}
    else:
        coefs = glm.coef()
        
    coefs = {k: {k1: v1 for k1, v1 in v.items() if abs(v1) > 0 and k1 != "Intercept"} for k, v in coefs.items()}
    
    rule_importance = dict()
    for k,v in coefs.items():
        rules_pd = pd.DataFrame.from_dict(v, orient = "index").reset_index() 
        if len(rules_pd) > 0:
            rules_pd.columns = ["variable", "coefficient"]
        rule_importance[k] = rules_pd

    # Convert paths to rules
    for k,v in rule_importance.items():
        class_rules = []
        if len(v) > 0:
            for i in v.variable:
                model_idx, tree_num, tree_class, path = _map_column_name(i, family, algorithm)
                tree = H2OTree(tree_models[model_idx], tree_num, tree_class = tree_class)
                class_rules = class_rules + [_tree_traverser(tree.root_node, path)]
        
            # Add rules and order by absolute coefficient
            v["rule"] = class_rules
            v["abs_coefficient"] = v["coefficient"].abs()
            v = v.loc[v.groupby(["rule"])["abs_coefficient"].idxmax()]  
            v = v.sort_values(by = "abs_coefficient", ascending = False)
            v = v.drop("abs_coefficient", axis = 1)
        
        rule_importance[k] = v
        
    if family != "multinomial":
        rule_importance = list(rule_importance.values())[0]
    
    return rule_importance
    
def _map_column_name(column_name, family, algorithm):
    """
    Take column name from paths frame and return the model_idx, tree_num, tree_class, and path 
    """
    if family == "binomial":
        if algorithm == "XGBoost":
            model_idx, tree_num, path = column_name.replace("tree_", "").replace("T", "").split(".")
            tree_class = int(0)
        else:
            model_idx, tree_num, tree_class, path = column_name.replace("tree_", "").replace("T", "").replace("C", "").split(".")
            tree_class = int(tree_class) - 1
    else:
        model_idx, tree_num, path = column_name.replace("tree_", "").replace("T", "").split(".")
        tree_class = None
        
    return int(model_idx), int(tree_num) - 1, tree_class, path

