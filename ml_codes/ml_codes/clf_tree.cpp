//
//  clf_tree.cpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/13.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#include "clf_tree.hpp"
#include <string>
#include <sstream>
using namespace std;

CLF_tree::CLF_tree_param::CLF_tree_param(){
    min_samples_split = 2;
    max_depth = 0;
};
/*
CLF_tree::CLF_tree_param::CLF_tree_param(CLF_tree_param& param_inp){
    min_samples_split = param_inp.min_samples_split;
    max_depth = param_inp.max_depth;

};
*/

CLF_tree::CLF_tree(CLF_tree_param param){
    set_param(param);
};

CLF_tree::CLF_tree(){
};

CLF_tree::~CLF_tree(){
};


void CLF_tree::set_param(CLF_tree_param param){
    this->param = param;
};


void CLF_tree::rc_growth_tree(vector<int>& data_idx){
    
    bool leaf_flag = true;
    idx_size_for_is_min_samples_split = data_idx.size();
    Node* split_node = tree.get_node();
    
    //終了条件
    if (!stop_condition()){
        //分割の作成
        Tree_Split split = make_split(data_idx);
        
        //分割が可能であれば、（moreもlessも要素が0でなければ）
        if(split.more_idx.size()>0&&split.less_idx.size()>0){
            
            // 分割値の設定
            split_node->split_value = split.split_value;
            split_node->feature_idx = split.feature_idx;
            
            // leaf_flagをfalseに設定
            leaf_flag = false;
            
            // 親ノードの情報を書き込み
            stringstream path_stream;
            path_stream.str("");
            path_stream << split_node->path;
            path_stream << "dep_"<<split_node->depth+1<<",";
            path_stream << "f_idx_"<<split_node->feature_idx<<",";
            path_stream << "spt_"<<split_node->split_value<<",";
            string path_common = path_stream .str();
            
            //more nodeへ
            tree.add_more_node();
            tree.to_more_node();
            tree.temp_node->path = path_common + "more" + "/";
            rc_growth_tree(split.more_idx);
            tree.to_parent_node();
            
            //less nodeへ
            tree.add_less_node();
            tree.to_less_node();
            tree.temp_node->path = path_common + "less" + "/";
            rc_growth_tree(split.less_idx);
            tree.to_parent_node();
        }
        
    }
    
    if(leaf_flag){
        // 葉の値を作成。
        make_leaf_value(data_idx, split_node);
    }
}

void CLF_tree::make_leaf_value(vector<int>& data_idx,Node* split_node){
    map<double,int> leaf_count_elem;
    for(auto leaf_iter = data_idx.cbegin();
        leaf_iter < data_idx.cend();
        leaf_iter++){
        leaf_count_elem[(*y_train)(*leaf_iter,0)] += 1;
    }
    
    double key = 0;
    int cnt=0;
    
    for (auto iter = leaf_count_elem.cbegin();iter != leaf_count_elem.cend();++iter){
        if (iter->second>cnt){
            key = iter->first;
            cnt = iter->second;
        }
    }
    
    split_node->leaf_value = key;
    
    stringstream path_stream;
    path_stream << split_node->path;
    path_stream <<"leaf_val->"<<split_node->leaf_value;
    split_node->path = path_stream.str();
}


CLF_tree::Tree_Split CLF_tree::make_split(vector<int>& data_idx){
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
    for(int c_idx = 0; c_idx < X_train->cols() ; ++c_idx){
        
        // 分割値を探索
        Eigen::MatrixXd feature_elems = X_train->col(c_idx);
        
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
                split.feature_idx = c_idx;
                split.split_value = split_point;
                split.split_score = split_score;
            }
            
        }
    }
    return split;
}


double CLF_tree::evaluation(vector<int>& more_idx, vector<int>& less_idx){
    return gini(more_idx,less_idx);
};

double CLF_tree::gini(vector<int>& more_idx, vector<int>& less_idx){
    //vector<int> more_idxとvector<int> less_idxより
    //gini不順度を算出する。
    
    map<double,double> more_count_elem, less_count_elem;//ラベルごとの出現数の連想配列
    double more_gini, less_gini;

    
    // more_idxでのy_trainのラベルごとの出現数のカウント
    for(auto more_iter = more_idx.cbegin();
        more_iter != more_idx.cend();
        more_iter++){
        more_count_elem[(*y_train)(*more_iter,0)] += 1;
    }
    
    // more_idxでのy_trainのラベルごとの出現数のカウント
    for(auto less_iter = less_idx.cbegin();
        less_iter < less_idx.cend();
        less_iter++){
        less_count_elem[(*y_train)(*less_iter,0)] += 1;
    }
    
    // more_idxでのgini不順度の計算
    more_gini = 1;
    for(auto more_count_iter = more_count_elem.cbegin();
        more_count_iter != more_count_elem.end();
        more_count_iter++){
        
        more_gini -= (more_count_iter->second / more_idx.size()) * (more_count_iter->second / more_idx.size());
    }
    
    // less_idxでのgini不順度の計算
    less_gini = 1;
    for(auto less_count_iter = less_count_elem.cbegin();
        less_count_iter != less_count_elem.end();
        less_count_iter++){
        
        less_gini -= (less_count_iter->second / less_idx.size()) * (less_count_iter->second / less_idx.size());
    }
    
    // more_gini, less_giniの荷重平均
    double gini = (more_gini * more_idx.size() + less_gini * less_idx.size())/(more_idx.size() + less_idx.size());
    
    return gini;
};


bool CLF_tree::stop_condition(){return is_max_depth()|is_min_samples_split();}
bool CLF_tree::is_max_depth(){return tree.get_node()->depth >= param.max_depth ? true : false;}
bool CLF_tree::is_min_samples_split(){return idx_size_for_is_min_samples_split < param.min_samples_split ? true : false;}

void CLF_tree::fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y){
    //木の初期化
    if(tree.node_list.size()>1){
        tree.del_all_nodes();
        tree.set_root_node();
    }
    // データの設定
    X_train = &(X);
    y_train = &(y);
    
    //インデックスの作成
    vector<int> data_idx;
    for(int i = 0; i < X.rows(); ++i){
        data_idx.push_back(i);
    }
    
    rc_growth_tree(data_idx);
    
}
double CLF_tree::one_pred(Eigen::MatrixXd& X_row){
    
    tree.to_root_node();
    
    while(tree.is_more_node()&&tree.is_less_node()){
        if(X_row(0,tree.get_node()->feature_idx) >= tree.get_node()->split_value){
            tree.to_more_node();
        }else{
            tree.to_less_node();
        }
    }
    
    return tree.get_node()->leaf_value;
};

Eigen::MatrixXd CLF_tree::predict(const Eigen::MatrixXd& X_test){
    
    Eigen::MatrixXd temp(X_test.rows(),1);
    
    for(int i = 0; i < X_test.rows(); ++i){
        Eigen::MatrixXd data_row = X_test.row(i);
        temp(i,0) = one_pred(data_row);
    }
    
    return temp;
    
};
