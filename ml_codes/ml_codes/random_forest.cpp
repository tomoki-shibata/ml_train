//
//  random_forest.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/29.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#include "random_forest.hpp"

///////////////////////////////////////////////
// Random_Forest_clf::RF_CLF_tree
///////////////////////////////////////////////

Random_Forest_clf::RF_CLF_tree::RF_CLF_tree():CLF_tree(){};


void Random_Forest_clf::RF_CLF_tree::set_forest(Random_Forest_clf* forest_address){
    forest_pointer = forest_address;
};


Random_Forest_clf::RF_CLF_tree::Tree_Split Random_Forest_clf::RF_CLF_tree::make_split(vector<int>& data_idx){
    //データを列のインデックス（c_idx）と分割値（split_val）で分割し、
    //evaluation()の出力(split_score)が最小となる時の
    //列のインデックス（c_idx）
    //分割値（split_val）
    //分割値以上の値のインデックス（more_idx）
    //分割値未満の値のインデックス（less_idx）
    //evaluation()の出力(split_score)
    //をsplitに格納して出力する。

    vector<int> more_idx;
    vector<int> less_idx;
    Tree_Split split;
    
    //特徴量でループして、分割を探索
    vector<int> col_list = forest_pointer->rand->col_sampling(forest_pointer->param.feature_sample_num,(int)X_train->cols());
    for(auto c_idx = col_list.cbegin(); c_idx != col_list.cend(); ++c_idx){
        
        // 分割値を探索
        Eigen::MatrixXd feature_elems = X_train->col(*c_idx);
        
        //ループを抜けるたために、サンプル数を総当たりする。
        for(auto split_idx_iter = data_idx.cbegin();
            split_idx_iter < data_idx.cend();
            split_idx_iter++){
            //idxのクリア
            more_idx.clear();
            less_idx.clear();
            
            // 分割値でデータを分割する。
            double split_point = feature_elems(*split_idx_iter,0);
            for(auto idx_iter = data_idx.cbegin();
                idx_iter != data_idx.cend();
                idx_iter++){
                
                //分割値のからの大小でサンプル（のインデックス）を分割
                if(split_point > feature_elems((*idx_iter),0)){
                    //分割値よりも小のケース
                    less_idx.push_back((*idx_iter));
                }else{
                    //分割値以上のケース
                    more_idx.push_back((*idx_iter));
                }
            }
            
            double split_score = evaluation(more_idx,less_idx);
            
            // split_scoreがsplit.split_scoreより小さい時、splitを更新する。
            if (split.split_score > split_score){
                split.more_idx = more_idx;
                split.less_idx = less_idx;
                split.feature_idx = *c_idx;
                split.split_value = split_point;
                split.split_score = split_score;
            }
            
        }
    }
    return split;
};

void Random_Forest_clf::RF_CLF_tree::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, vector<int>& data_idx){
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

///////////////////////////////////////////////
// Rand
///////////////////////////////////////////////

vector<int> Random_Forest_clf::Rand::row_sampling(int sample_num, int rows){
    //重複を許してサンプリング
    vector<int> row_idx;
    for(int i = 0;i < sample_num;++i){
        row_idx.push_back(mt_rand()%rows);
    }
    return row_idx;
};


vector<int> Random_Forest_clf::Rand::col_sampling(int sample_num, int cols){
    //重複を許さずにサンプリング
    vector<int> col_idx(sample_num);
    std::vector<int> col_num(cols);
    for (int i = 0; i < cols; i++) {
        col_num[i] = i;
    }
    
    int t = 0;
    int m = 0;
    while (m < sample_num) {
        double u = ((double)mt_rand())/0xffffffff;//0~1の一様乱数を取得
        if ((cols - t) * u >= (sample_num - m)) {
            t++;
        } else {
            col_idx[m] = col_num[t];
            t++;
            m++;
        }
    }
    return col_idx;
};

///////////////////////////////////////////////
// Random_Forest_clf_param
///////////////////////////////////////////////

Random_Forest_clf::Random_Forest_clf_param::Random_Forest_clf_param(){
    feature_sample_num = 0;
    random_seed = 0;
};


Random_Forest_clf::Random_Forest_clf_param::Random_Forest_clf_param(Random_Forest_clf_param& param_inp){
    feature_sample_num = param_inp.feature_sample_num;
    random_seed = param_inp.random_seed;
    tree_param.min_samples_split = param_inp.tree_param.min_samples_split;
    tree_param.max_depth = param_inp.tree_param.max_depth;
    
};


///////////////////////////////////////////////
// Random_Forest_clf
///////////////////////////////////////////////
Random_Forest_clf::Random_Forest_clf(){
    rand = &rand_instance;
};

Random_Forest_clf::Random_Forest_clf(Random_Forest_clf_param& inp_param){
    
    set_param(inp_param);
    rand = &rand_instance;

};


void Random_Forest_clf::set_param(Random_Forest_clf_param& inp_param){
    
    param.estimators_num = inp_param.estimators_num;
    param.feature_sample_num = inp_param.feature_sample_num;
    param.random_seed = inp_param.random_seed;
    param.tree_param.min_samples_split = inp_param.tree_param.min_samples_split;
    param.tree_param.max_depth = inp_param.tree_param.max_depth;
    
};


void Random_Forest_clf::fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y){
    
    //乱数初期化
    rand->mt_rand.seed(param.random_seed);
    
    X_train = &X;
    y_train = &y;
    
    forest.clear();
    forest.resize(param.estimators_num);
    
    for(RF_CLF_tree& iter:forest){
        vector<int> data_idx = rand->row_sampling((int)X_train->rows(),(int)X_train->rows());
        
        iter.set_param(param.tree_param);
        iter.set_forest(this);
        iter.fit(*X_train, *y_train, data_idx);
    }
    
}


Eigen::MatrixXd Random_Forest_clf::predict(const Eigen::MatrixXd& X_test){
    Eigen::MatrixXd temp = Eigen::MatrixXd(X_test.rows(),1);
    vector<map<double,int>> result_count(X_test.rows());
    
    //RF_CLF_tree.pred()を使って、出力を出す。

    for(RF_CLF_tree& iter:forest){
        temp = iter.predict(X_test);
        
        for(int i = 0; i < temp.rows(); ++i){
            result_count[i][temp(i,0)]+=1;
        }
    }
    
    Eigen::MatrixXd result(temp.rows(),1);
    
    for(int i = 0; i < temp.rows(); ++i){
        double key = 0;
        int cnt = 0;

        for(auto result_iter = result_count[i].cbegin();result_iter != result_count[i].cend();result_iter++){
            if(result_iter->second > cnt){
                key = result_iter->first;
                cnt = result_iter->second;
            }
        }
        result(i, 0) = key;
    }
    
    return result;
};



