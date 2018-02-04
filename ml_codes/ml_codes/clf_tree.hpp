//
//  clf_tree.hpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/13.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//


#ifndef clf_tree_hpp
#define clf_tree_hpp

#include <iostream>
#include <vector>
#include <utility>
#include <eigen3/Eigen/Dense>
#include <map>
#include "tree.hpp"
#include "gtest/gtest.h"
using namespace std;

class CLF_tree{
//クラス内クラスの定義
public:
    struct CLF_tree_param{
        int min_samples_split;
        int max_depth;
        CLF_tree_param();
        //CLF_tree_param(CLF_tree_param& param_inp);
    };
    
protected:
    struct Tree_Split {
        vector<int> more_idx;
        vector<int> less_idx;
        int feature_idx;
        double split_value;
        double split_score = DBL_MAX;
        
    };
    
//メンバの定義
public:
    CLF_tree(CLF_tree_param param);
    CLF_tree();
    void set_param(CLF_tree_param param);
    void fit(const Eigen::MatrixXd& X_train, const Eigen::MatrixXd& y_train);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X_test);

protected:
    Tree tree;
    const Eigen::MatrixXd* X_train;
    const Eigen::MatrixXd* y_train;
    
    //木のパラメータ
    CLF_tree_param param;
    
    // ノードの分割
    virtual Tree_Split make_split(vector<int>& data_idx);
    virtual double gini(vector<int>& more_idx, vector<int>& less_idx);
    
    // 木の成長条件
    virtual bool is_max_depth();
    virtual bool is_min_samples_split();
    long idx_size_for_is_min_samples_split;
    
    // 木の成長
    virtual void rc_growth_tree(vector<int>& data_idx);
    virtual double evaluation(vector<int>& more_idx, vector<int>& less_idx);
    virtual bool stop_condition();
    
    //一回予測
    virtual int one_pred(Eigen::MatrixXd& X_row);
    
    //テストクラスのフレンド宣言
    friend class ClfTreeTest;
    FRIEND_TEST(ClfTreeTest, max_depth);
    FRIEND_TEST(ClfTreeTest, min_samples_split);
    FRIEND_TEST(ClfTreeTest, stop_condition);
    FRIEND_TEST(ClfTreeTest, gini);
    FRIEND_TEST(ClfTreeTest, make_split);
    FRIEND_TEST(ClfTreeTest, rc_growth_tree);
    FRIEND_TEST(ClfTreeTest, fit);
    FRIEND_TEST(ClfTreeTest, one_pred);
    FRIEND_TEST(ClfTreeTest, predict);

};




#endif /* clf_tree_hpp */
