//
//  LightGBM.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/02/07.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#include "LightGBM.hpp"

using namespace std;

///////////////////////////////////////////////
// LightGBM_clf::LightGBM_CLF_tree
///////////////////////////////////////////////
LightGBM_clf::LightGBM_CLF_tree::LightGBM_CLF_tree():CLF_tree(){};
LightGBM_clf::LightGBM_CLF_tree::~LightGBM_CLF_tree(){};


void LightGBM_clf::LightGBM_CLF_tree::set_forest(LightGBM_clf* forest_address){
    forest = forest_address;
}


double LightGBM_clf::LightGBM_CLF_tree::evaluation(vector<int>& more_idx, vector<int>& less_idx){
    return neg_variance_gain(more_idx, less_idx);
}


double LightGBM_clf::LightGBM_CLF_tree::neg_variance_gain(vector<int>& more_idx, vector<int>& less_idx){
    //make_splitでは最小化を行うため、負のゲインを返す。
    double more_gain, less_gain, total_gain;
    more_gain = less_gain = total_gain = 0;
    
    for(const auto& idx: more_idx){
        more_gain += forest->neg_factored_gradient[this_tree_label][idx];
    }
    more_gain *= more_gain;
    more_gain /= more_idx.size();
    
    for(const auto& idx: less_idx){
        less_gain += forest->neg_factored_gradient[this_tree_label][idx];
    }
    less_gain *= less_gain;
    less_gain /= less_idx.size();
    
    total_gain = more_gain + less_gain;
    //ここで割り込むのは分割目的では無駄だが、一応定義に従って計算する。
    total_gain /= (more_idx.size() + less_idx.size());
    
    //最大化問題を最小化問題に変更するために、マイナスを掛ける。
    return -total_gain;
    
}


void LightGBM_clf::LightGBM_CLF_tree::make_leaf_value(vector<int>& data_idx, Node *split_node){
    split_node->leaf_value = leaf_solver(data_idx, this_tree_label);
    
    stringstream path_stream;
    path_stream << split_node->path;
    path_stream <<"leaf_val->"<<split_node->leaf_value;
    split_node->path = path_stream.str();
}


double LightGBM_clf::LightGBM_CLF_tree::leaf_solver(vector<int>& data_idx, double tree_label){
    //それぞれの葉には、gamma = Δf(x)を学習させる。
    //このため、gammaの範囲をデフォルトで-2~2の範囲とする。
    double temp_loss, loss, temp_gamma, gamma;
    loss = DBL_MAX;
    gamma = 0;
    
    for(temp_gamma = forest->param.min_gamma;
        temp_gamma < forest->param.max_gamma + forest->param.gamma_precision/2;//計算嬢多少precisionが小さくなってもmaxまで計算するようにする。
        temp_gamma += forest->param.gamma_precision){
        
        temp_loss = loss_func_for_leaf(data_idx,tree_label,temp_gamma);
  
        // 最小値を探索
        if (loss > temp_loss){
            loss = temp_loss;
            gamma = temp_gamma;
        }
    }
    
    return gamma;
    
}


double LightGBM_clf::LightGBM_CLF_tree::loss_func_for_leaf(vector<int>& data_idx, double tree_label, double gamma){
    double denominator;
    double loss;
    loss = 0;
    
    for(const auto& row_idx:data_idx){
        denominator = 0;
        // 分母を作成
        for(const auto& cash:forest->cash_result){
            denominator += exp(cash.second(row_idx,0));
        }
        // gammaがかかるf(x)を除いて、gannmaを足して指数関数に入力する計算で分母を完成。
        denominator -= exp(forest->cash_result[tree_label](row_idx,0));
        denominator += exp(forest->cash_result[tree_label](row_idx,0) + gamma);
        
        loss += log(denominator);
        
        //分子をラベルで場合分けして（ラベルが一致する時はgammaを乗じる。）作成する。
        if((*y_train)(row_idx,0) == tree_label){
            //ラベルが木と一致するときはgammaを加える。
            loss += -(forest->cash_result[tree_label](row_idx,0) + gamma);
        }else{
            //ラベルが一致しない時はそのまま
            loss += -forest->cash_result[tree_label](row_idx,0);
        }
    }
    
    return loss;
}


void LightGBM_clf::LightGBM_CLF_tree::set_leaf_value_on_root(double leaf_value){
    stringstream ss;
    tree.root_node->leaf_value = leaf_value;
    ss << "/leaf_val->" << leaf_value;
    tree.root_node->path = ss.str();
    tree.root_node->more_node = nullptr;
    tree.root_node->less_node = nullptr;
};


void LightGBM_clf::LightGBM_CLF_tree::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, vector<int>& data_idx){
    //木の初期化
    if(tree.node_list.size()>1){
        tree.del_all_nodes();
        tree.set_root_node();
    }
    // データの設定
    X_train = &(X);
    y_train = &(y);
    
    rc_growth_tree(data_idx);
    
}


/////////////////////////////////////////////
// LightGBM_clf::Rand
/////////////////////////////////////////////

map<double,vector<int>> LightGBM_clf::Rand::GOS_sampling(map<double,vector<double>>& neg_gradient, double high_gradient_sampling_ratio, double low_gradient_sampling_ratio){
    map<double,vector<int>> temp_sampled_idx; //サンプルされたインデックス
    
    // factorを掛けるサイズを計算。
    int target_data_size = (int)neg_gradient.cbegin()->second.size();
    int const_sample_size = (int)(target_data_size * high_gradient_sampling_ratio);
    int random_sample_size = (int)(target_data_size * low_gradient_sampling_ratio);
    
    // ラベル毎に処理
    for(const auto& label_and_neg_grad:neg_gradient){
        double label = label_and_neg_grad.first;
        vector<double> abs_grad;
        
        // gradientの絶対値を取得
        for(const auto& grad : label_and_neg_grad.second){
            abs_grad.push_back(grad > 0 ? grad : -grad);
        };
        
        // 降順順となるようにインデックスをソート
        //インデックスの作成
        vector<int> grad_idx(abs_grad.size());
        iota(grad_idx.begin(), grad_idx.end(), 0);
        
        //降順ソート
        sort(grad_idx.begin(),grad_idx.end(),
             [&](int x, int y){return abs_grad[x] > abs_grad[y];});
        
        // gradientが大きなものだけをサンプル
        for(int i = 0; i < const_sample_size; ++i){
            temp_sampled_idx[label].push_back(grad_idx[i]);
        }
        
        // gradientが小さなものを非復元ランダムサンプル
        const vector<int> low_grad_idx = random_sampling_without_replacement(random_sample_size, target_data_size - const_sample_size, const_sample_size);
        
        // low gradientの配列を連結
        //temp_sampled_idx[label].insert(temp_sampled_idx[label].end(),low_grad_idx.begin(),low_grad_idx.end());
        for(auto& idx : low_grad_idx){
            temp_sampled_idx[label].push_back(grad_idx[idx]);
        }
        
    }
    
    return temp_sampled_idx;
    
};


const vector<int> LightGBM_clf::Rand::random_sampling_without_replacement(int sample_num, int target_data_num, int return_base){
    // 重複を許さずにサンプリング
    //
    vector<int> sampled_idx;
    std::vector<int> target_data_idx(target_data_num);
    
    // サンプル対象の列を作成。
    for (int i = 0; i < target_data_num; i++) {
        target_data_idx[i] = return_base + i;
    }
    
    int t = 0;
    int m = 0;
    while (m < sample_num) {
        double u = ((double)mt_rand())/0xffffffff;//0~1の一様乱数を取得
        if ((target_data_num - t) * u >= (sample_num - m)) {
            t++;
        } else {
            sampled_idx.push_back(target_data_idx[t]);
            t++;
            m++;
        }
    }
    return sampled_idx;
    
};


///////////////////////////////////////////////
// LightGBM_clf::LightGBM_clf_param
///////////////////////////////////////////////

LightGBM_clf::LightGBM_clf_param::LightGBM_clf_param(){
    estimators_num = 1;
    random_seed = 0;
    high_gradient_sampling_ratio = 0.5;
    low_gradient_sampling_ratio = 0.5;
    max_gamma = 2.0;
    min_gamma = -2.0;
    gamma_precision = 0.1;

};


///////////////////////////////////////////////
// LightGBM_clf
///////////////////////////////////////////////
LightGBM_clf::LightGBM_clf(){
    rand = &rand_instance;
};


LightGBM_clf::LightGBM_clf(LightGBM_clf_param& inp_param){
    set_param(inp_param);
    rand = &rand_instance;
};


void LightGBM_clf::set_param(LightGBM_clf_param& inp_param){
    param.estimators_num = inp_param.estimators_num;
    param.random_seed = inp_param.random_seed;
    param.high_gradient_sampling_ratio = inp_param.high_gradient_sampling_ratio;
    param.low_gradient_sampling_ratio = inp_param.low_gradient_sampling_ratio;
    
    param.max_gamma = inp_param.max_gamma;
    param.min_gamma = inp_param.min_gamma;
    param.gamma_precision = inp_param.gamma_precision;
    
    param.tree_param.min_samples_split = inp_param.tree_param.min_samples_split;
    param.tree_param.max_depth = inp_param.tree_param.max_depth;
};


map<double,Eigen::MatrixXd> LightGBM_clf::make_cash_result(const Eigen::MatrixXd& X, int generation){
    map<double,Eigen::MatrixXd> cash; //木の出力のキャッシュ
    
    // labelでループ
    for(auto& labeled_trees:forest){
        double label = labeled_trees.first;
        cash[label] = Eigen::MatrixXd::Zero(X.rows(),1);
        
        // 世代でループ
        vector<LightGBM_CLF_tree> &trees = labeled_trees.second;
        for(int i = 0;i <= generation;++i){
            cash[label] += trees[i].predict(X);
            }
    }
    
    return cash;
    
};


map<double,vector<double>> LightGBM_clf::neg_gradient_cross_entropy(const Eigen::MatrixXd& y){
    map<double,vector<double>> temp_neg_gradient;// ラベル毎×入力データ毎のgradient
    double calc_temp,denomi;
    
    for(const auto& label_and_cash:cash_result){
        //temp_neg_gradient[label_and_cash.first] = vector<double>{};
        for(int row_idx = 0; row_idx < y.rows();++row_idx){//行でループ
            calc_temp = denomi = 0;
            // 確率の分母作成
            for(const auto& cash_for_calc:cash_result){
                denomi += exp(cash_for_calc.second(row_idx, 0));
            }
            //確率作成
            calc_temp = exp(label_and_cash.second(row_idx,0))/denomi;
            
            //勾配作成
            if(label_and_cash.first == y(row_idx,0)){
                calc_temp = 1 - calc_temp;
            }else{
                calc_temp = -calc_temp;
            }
            
            // 負の勾配化&データ格納
            temp_neg_gradient[label_and_cash.first].push_back(-calc_temp);
        }
    }
    return  temp_neg_gradient;
}


map<double,vector<double>> LightGBM_clf::factor_neg_gradient(double high_gradient_sampling_ratio, double low_gradient_sampling_raito){
    map<double,vector<double>> temp_factored_gradient;// ラベル毎×入力データ毎のfactored gradient
    
    // 一旦、temp_factored_gradientにneg_gradientを代入
    temp_factored_gradient = neg_gradient;
    
    // factorを掛けるサイズを計算。
    int data_row_size = (int)temp_factored_gradient.cbegin()->second.size();
    int factor_size = data_row_size - (int)(data_row_size * high_gradient_sampling_ratio);
    
    // ラベル毎に処理
    for(const auto& label_and_neg_grad:neg_gradient){
        double label = label_and_neg_grad.first;
        vector<double> abs_grad;
        
        // gradientの絶対値を取得
        for(const auto& grad : label_and_neg_grad.second){
            abs_grad.push_back(grad > 0 ? grad : -grad);
        };
        
        // 昇順となるようにインデックスをソート
        vector<int> grad_idx(abs_grad.size());
        iota(grad_idx.begin(), grad_idx.end(), 0);
        sort(grad_idx.begin(),grad_idx.end(), [&](int x, int y){return abs_grad[x] < abs_grad[y];});
        
        //　factor_sizeになるまで、grad_idxのインデクスを回す。
        double factor = (1 - high_gradient_sampling_ratio)/low_gradient_sampling_raito;
        for(int i = 0; i < factor_size; ++i){
            temp_factored_gradient[label][grad_idx[i]] = temp_factored_gradient[label][grad_idx[i]] * factor;
        };
        
    }
    
    return temp_factored_gradient;
}


tuple<vector<double>, map<double, int>> LightGBM_clf::count_label(const Eigen::MatrixXd& y_train){
    map<double, int> label_count; // ラベル毎の出現回数
    vector<double> temp_label_list; // ラベル一覧
    
    // データ全件をチェック
    for(int i = 0; i < y_train.rows(); ++i){
        label_count[y_train(i,0)] += 1;
    }
    
    // mapのキーからラベル一覧を作成
    for(const auto& labe_and_count : label_count){
        temp_label_list.push_back(labe_and_count.first);
    }
    
    return make_tuple(temp_label_list,label_count);
};

void LightGBM_clf::init_0_generation_trees(vector<double>& label_list, map<double,int>& label_count){
    // 初期値計算用の最小値
    int min_count;
    min_count = INT_MAX;
    for(const double label : label_list){
        if(min_count > label_count[label]){
            min_count = label_count[label];
        }
    }
    
    // クリア
    forest.clear();
    for(const double label : label_list){
        //初期化
        forest[label].resize(param.estimators_num);
        
        // 0_generation_treesの設定
        forest[label][0].forest_generation = 0;
        forest[label][0].this_tree_label = label;
        forest[label][0].set_forest(this);
        
        // 木の初期予測をlog(木のラベルのデータ数/最小のラベルのデータ数)とする。
        forest[label][0].set_leaf_value_on_root(log(((double)label_count[label])/min_count));
    }
};


void LightGBM_clf::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y){
    
    //乱数初期化
    rand->mt_rand.seed(param.random_seed);
    
    // データセット
    X_train = &X;
    y_train = &y;
    
    // データ数の取得
    data_count = (int)X_train->rows();
    
    //ラベルの整理
    tuple<vector<double>, map<double,int>> label_tuple = count_label(*y_train);
    label_list = get<0>(label_tuple);
    label_count = get<1>(label_tuple);

    // 木の初期化
    init_0_generation_trees(label_list, label_count);
    
    // 学習部分
    for(int i = 1; i < param.estimators_num; i++){
        // キャッシュ作成
        cash_result = make_cash_result(*X_train,i);
        // 重み付きゲインの計算
        neg_gradient = neg_gradient_cross_entropy(*y_train);
        neg_factored_gradient = factor_neg_gradient(param.high_gradient_sampling_ratio, param.low_gradient_sampling_ratio);
        // GOS-sampling
        map<double,std::vector<int>> data_idx_map = rand->GOS_sampling(neg_gradient, param.high_gradient_sampling_ratio, param.low_gradient_sampling_ratio);
        
        for(const double label : label_list){
            forest[label][i].forest_generation = i;
            forest[label][i].this_tree_label = label;
            forest[label][i].set_forest(this);
            forest[label][i].set_param(param.tree_param);
            forest[label][i].fit(*X_train, *y_train, data_idx_map[label]);
            
        }
    }
    
};


Eigen::MatrixXd LightGBM_clf::predict(const Eigen::MatrixXd& X_test){
    
    map<double, Eigen::MatrixXd> prob_result;
    Eigen::MatrixXd denomi;
    Eigen::MatrixXd max_prob_label;
    
    // Matrix初期化
    for(auto label : label_list){
        prob_result[label] = Eigen::MatrixXd::Zero(X_test.rows(),1);
    }
    denomi = Eigen::MatrixXd::Zero(X_test.rows(),1);
    max_prob_label = Eigen::MatrixXd::Zero(X_test.rows(),1);
    
    // 確率計算（分子）
    for(int i = 0; i < param.estimators_num; i++){
        for(const double label : label_list){
           prob_result[label] += forest[label][i].predict(X_test);
        }
    }
    
    // 確率計算（分母）
    for(const double label : label_list){
        denomi += prob_result[label].array().exp().matrix();
    }
    
    //確率計算（分子/分母）
    for(const double label : label_list){
        prob_result[label] = (prob_result[label].array() / denomi.array()).matrix();
    }
    
    current_probs_map = prob_result;
    
    //最大確率のラベルの取得
    for(int i = 0;i < X_test.rows(); ++i){
        //ラベルの初期化
        max_prob_label(i,0) = (*label_list.begin());
        for(const double label : label_list){
            if(current_probs_map[max_prob_label(i,0)](i,0) < current_probs_map[label](i,0)){
                max_prob_label(i,0) = label;
            }
        }
    }
    
    return max_prob_label;
    
};


