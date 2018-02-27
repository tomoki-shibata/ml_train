//
//  light_gbm_test.h
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/02/12.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef light_gbm_test_h
#define light_gbm_test_h

#include "gtest/gtest.h"
#include "LightGBM.hpp"
#include <array>
using namespace std;

namespace{
    class LightGBMTest : public ::testing::Test{
    protected:
        LightGBMTest(){}
        virtual ~LightGBMTest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
    };
}


TEST_F(LightGBMTest, count_label)
{
    LightGBM_clf gbm;
    Eigen::MatrixXd y_train;
    y_train = Eigen::MatrixXd(10,1);
    y_train << 1, 2, 2.1, 1, 2, 2, 2.1, 1, 1, 2;
    
    tuple<vector<double>, map<double,int>> label_tuple = gbm.count_label(y_train);
    vector<double> label_list = get<0>(label_tuple);
    map<double,int> label_count = get<1>(label_tuple);
    
    EXPECT_EQ(1 , label_list[0]);
    EXPECT_EQ(2 , label_list[1]);
    EXPECT_EQ(2.1 , label_list[2]);
    EXPECT_EQ(4 , label_count[1]);
    EXPECT_EQ(4 , label_count[2]);
    EXPECT_EQ(2 , label_count[2.1]);
    
}


TEST_F(LightGBMTest, set_leaf_value_on_root)
{
    LightGBM_clf::LightGBM_CLF_tree tree;
    
    double leaf_val;
    Eigen::MatrixXd X_train, y_pred;
    
    leaf_val = 10;
    tree.set_leaf_value_on_root(leaf_val);
    
    X_train = Eigen::MatrixXd(5,2);
    X_train<<
    1       ,1,
    1000    ,-500,
    3.1419  , -666,
    0       ,0.001,
    -2      ,-4;
    
    y_pred = tree.predict(X_train);
    
    for(int i = 0; i < X_train.rows() ;++i){
        EXPECT_EQ(leaf_val, y_pred(i,0));
    }
    
    leaf_val = -314.1952;
    tree.set_leaf_value_on_root(leaf_val);
    
    y_pred = tree.predict(X_train);
    
    for(int i = 0; i < X_train.rows() ;++i){
        EXPECT_EQ(leaf_val, y_pred(i,0));
    }
}


TEST_F(LightGBMTest, init_0_generation_trees)
{
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train, y_train, y_pred;
    int estimators_num;
    
    estimators_num = 2;
    
    y_train = Eigen::MatrixXd(10,1);
    y_train << 1, 2, -2.1, 1, 2, 2, -2.1, 1, 1, 1;
 
    // 1    -> 5回
    // 2    -> 3回
    // −2.1 -> 2回
 
    map<double,double> leaf_vals{
        {1,log(5.0/2)},
        {2,log(3.0/2)},
        {2.1,log(2.0/2)}
    };

    
    X_train = Eigen::MatrixXd(5,2);
    X_train<<
    1       ,1,
    1000    ,-500,
    3.1419  , -666,
    0       ,0.001,
    -2      ,-4;
    

    tuple<vector<double>, map<double,int>> label_tuple = gbm.count_label(y_train);
    gbm.label_list = get<0>(label_tuple);
    gbm.label_count = get<1>(label_tuple);
    
    
    //初期化
    gbm.init_0_generation_trees(gbm.label_list, gbm.label_count);
    
    for(const double label : gbm.label_list){
        EXPECT_EQ(0, gbm.forest[label][0].forest_generation);
        EXPECT_EQ(label, gbm.forest[label][0].this_tree_label);
        EXPECT_EQ(&gbm, gbm.forest[label][0].forest);
        y_pred = gbm.forest[label][0].predict(X_train);
        
        for(int i = 0; i < X_train.rows() ;++i){
            EXPECT_EQ(leaf_vals[label], y_pred(i,0));
        }
    }
}



TEST_F(LightGBMTest, make_cash_result){
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train, y_train, y_pred;
    int estimators_num;
    
    estimators_num = 2;
    
    y_train = Eigen::MatrixXd(10,1);
    y_train << 1, 2, -2.1, 1, 2, 2, -2.1, 1, 1, 1;
 
     //1    -> 5回
     //2    -> 3回
     //−2.1 -> 2回
 
    map<double,double> leaf_vals{
        {1,log(5.0/2)},
        {2,log(3.0/2)},
        {2.1,log(2.0/2)}
    };
    
    
    X_train = Eigen::MatrixXd(5,2);
    X_train<<
    1       ,1,
    1000    ,-500,
    3.1419  , -666,
    0       ,0.001,
    -2      ,-4;
    
    
    tuple<vector<double>, map<double,int>> label_tuple = gbm.count_label(y_train);
    
    gbm.label_list = get<0>(label_tuple);
    gbm.label_count = get<1>(label_tuple);
    LightGBM_clf::LightGBM_clf_param param;
    param.estimators_num = 1;
    gbm.set_param(param);
    
    //初期化
    gbm.init_0_generation_trees(gbm.label_list, gbm.label_count);
    
    gbm.cash_result = gbm.make_cash_result(X_train,1);
    
    for(auto& label_and_cash:gbm.cash_result){
        double label = label_and_cash.first;
        Eigen::MatrixXd cash_data = label_and_cash.second;
        
        for(int i = 0; i<cash_data.rows(); ++i){
            EXPECT_EQ(leaf_vals[label],cash_data(i,0));
        }
    }
     
    
    //2回目
    X_train = Eigen::MatrixXd(8,2);
    X_train <<
    7,   0,
    5, 1.9,
    6,-2.6,
    3, 0.8,
    3, 1.2,
    3,-1.3,
    4, 2.0,
    2, 4.1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 1,1,1,0,0,0,0,0;
    
    vector<int> idx{0,1,2,3,4,5,6,7};
    
    label_tuple = gbm.count_label(y_train);
    gbm.label_list = get<0>(label_tuple);
    gbm.label_count = get<1>(label_tuple);
    gbm.param.estimators_num = 2;
    gbm.forest.clear();
    gbm.init_0_generation_trees(gbm.label_list, gbm.label_count);
    
    // label = 0
    gbm.forest[gbm.label_list[0]][1].forest_generation = 1;
    gbm.forest[gbm.label_list[0]][1].this_tree_label = gbm.label_list[0];
    gbm.forest[gbm.label_list[0]][1].set_forest(&gbm);
    gbm.forest[gbm.label_list[0]][1].tree.temp_node->feature_idx=0;
    gbm.forest[gbm.label_list[0]][1].tree.temp_node->split_value=5;
    gbm.forest[gbm.label_list[0]][1].tree.add_more_node();
    gbm.forest[gbm.label_list[0]][1].tree.to_more_node();
    gbm.forest[gbm.label_list[0]][1].tree.temp_node->leaf_value=2;
    gbm.forest[gbm.label_list[0]][1].tree.to_parent_node();
    gbm.forest[gbm.label_list[0]][1].tree.add_less_node();
    gbm.forest[gbm.label_list[0]][1].tree.to_less_node();
    gbm.forest[gbm.label_list[0]][1].tree.temp_node->leaf_value=-2;
    
    
    //label = 1
    gbm.forest[gbm.label_list[1]][1].forest_generation = 1;
    gbm.forest[gbm.label_list[1]][1].this_tree_label = gbm.label_list[1];
    //gbm.neg_factored_gradient[gbm.label_list[0]] = vector<double>{1,1,1,1,1,1,1,1};
    gbm.forest[gbm.label_list[1]][1].set_forest(&gbm);
    //gbm.forest[gbm.label_list[1]][1].fit(X_train, y_train, idx);
    gbm.forest[gbm.label_list[1]][1].tree.temp_node->leaf_value=10;
    
    
    gbm.cash_result = gbm.make_cash_result(X_train,1);
    
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](0,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](1,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](2,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](3,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](4,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](5,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](6,0));
    EXPECT_EQ(log(5.0/3),gbm.cash_result[0](7,0));
    
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](0,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](1,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](2,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](3,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](4,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](5,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](6,0));
    EXPECT_EQ(log(3.0/3),gbm.cash_result[1](7,0));
    
    
    gbm.cash_result = gbm.make_cash_result(X_train,2);
    
    EXPECT_EQ(log(5.0/3)+2,gbm.cash_result[0](0,0));
    EXPECT_EQ(log(5.0/3)+2,gbm.cash_result[0](1,0));
    EXPECT_EQ(log(5.0/3)+2,gbm.cash_result[0](2,0));
    EXPECT_EQ(log(5.0/3)-2,gbm.cash_result[0](3,0));
    EXPECT_EQ(log(5.0/3)-2,gbm.cash_result[0](4,0));
    EXPECT_EQ(log(5.0/3)-2,gbm.cash_result[0](5,0));
    EXPECT_EQ(log(5.0/3)-2,gbm.cash_result[0](6,0));
    EXPECT_EQ(log(5.0/3)-2,gbm.cash_result[0](7,0));
    
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](0,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](1,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](2,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](3,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](4,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](5,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](6,0));
    EXPECT_EQ(log(3.0/3)+10,gbm.cash_result[1](7,0));
    
}


TEST_F(LightGBMTest, neg_gradient_cross_entropy){
    LightGBM_clf gbm;
    Eigen::MatrixXd y_train,temp;
    
    y_train = Eigen::MatrixXd(5,1);
    y_train << 1, 1, 2, 2, -2.1;
    temp = Eigen::MatrixXd(5,1);
    temp<<log(10),log(0.1),log(1),log(3),log(50);
    gbm.cash_result[1] = temp;
    temp<<log(5),log(0.05),log(1.5),log(3),log(50);
    gbm.cash_result[2] = temp;
    temp<<log(1),log(0.01),log(2.5),log(3),log(100);
    gbm.cash_result[-2.1] = temp;
    
    map<double,vector<double>> neg_grad = gbm.neg_gradient_cross_entropy(y_train);
    

    EXPECT_EQ(round(1000*(-6.0/16)),round(1000*neg_grad[1][0]));// -(1-10/16) = -0.375
    EXPECT_EQ(round(1000*(-6.0/16)),round(1000*neg_grad[1][1]));// -(1-0.1/0.16) = -0.375
    EXPECT_EQ(round(1000*1.0/5),round(1000*neg_grad[1][2]));// -(-1/5) = 0.2
    EXPECT_EQ(round(1000*1.0/3),round(1000*neg_grad[1][3]));//-(-3/9) = 0.333333
    EXPECT_EQ(round(1000*1.0/4),round(1000*neg_grad[1][4]));// -(-50/200) = 0.25
    
    EXPECT_EQ(round(100*5.0/16),round(100*neg_grad[2][0]));// -(-5/16) = 0.3125
    EXPECT_EQ(round(1000*5.0/16),round(1000*neg_grad[2][1]));// -(-0.05/0.16) = -0.375
    EXPECT_EQ(-round(1000*0.7),round(1000*neg_grad[2][2]));// -(1-1.5/5) = -0.7
    EXPECT_EQ(-round(1000*2.0/3),round(1000*neg_grad[2][3]));//-(1-3/9) = 0.666667
    EXPECT_EQ((1000*1.0/4),round(1000*neg_grad[2][4]));// -(-50/200) = 0.25
    
    EXPECT_EQ(round(1000*1.0/16),round(1000*neg_grad[-2.1][0]));// -(-1/16) = 0.0625
    EXPECT_EQ(round(1000*1.0/16),round(1000*neg_grad[-2.1][1]));// -(-0.01/0.16) = 0.0625
    EXPECT_EQ(round(1000*1.0/2),round(1000*neg_grad[-2.1][2]));// -(-2.5/5) = 0.5
    EXPECT_EQ(round(1000*1.0/3),round(1000*neg_grad[-2.1][3]));//-(-3/9) = 0.333333
    EXPECT_EQ(-round(1000*1.0/2),round(1000*neg_grad[-2.1][4]));// -(1-100/200) =-0.5
 
}

TEST_F(LightGBMTest, factor_neg_gradient){
    LightGBM_clf gbm;
    vector<double> a{0,1,2,3,4,5,6,7,8,9};
    vector<double> a_ans{0,2,4,6,8,10,12,14,8,9};
    gbm.neg_gradient[1] = a;
    
    vector<double> b{-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4};
    vector<double> b_ans{-0.5, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.4};
    gbm.neg_gradient[2] = b;
    
    vector<double> c{10,20,30,40,50,40,30,20,10,0};
    vector<double> c_ans{20,40,60,80,50,40,60,40,20,0};
    gbm.neg_gradient[-2.1] = c;
    
    map<double, vector<double>> f_grad = gbm.factor_neg_gradient(0.2, 0.4); //factor = 0.8/0.4 =2
    
    for(int i = 0;i < f_grad.begin()->second.size();++i){
        EXPECT_EQ(a_ans[i],f_grad[1][i]);
        EXPECT_EQ(b_ans[i],f_grad[2][i]);
        EXPECT_EQ(c_ans[i],f_grad[-2.1][i]);
    }
    
    vector<double> a_ans2{0,2,4,6,8,10,12,14,8,9};
    vector<double> b_ans2{-0.5, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.4};
    vector<double> c_ans2{20,40,60,80,50,40,60,40,20,0};
    
    f_grad = gbm.factor_neg_gradient(0.25, 0.375); //factor = 0.75/0.375 = 2
    
    for(int i = 0;i < f_grad.begin()->second.size();++i){
        EXPECT_EQ(a_ans2[i],f_grad[1][i]);
        EXPECT_EQ(b_ans2[i],f_grad[2][i]);
        EXPECT_EQ(c_ans2[i],f_grad[-2.1][i]);
    }
}



TEST_F(LightGBMTest, neg_variance_gain){
    LightGBM_clf gbm;
    gbm.forest[-2.1].resize(1);
    gbm.forest[-2.1][0].this_tree_label = -2.1;
    gbm.forest[-2.1][0].set_forest(&gbm);
    
    vector<double> grad{1,1,1,2,2};
    vector<int> more_idx{0,1,2};
    vector<int> less_idx{3,4};
    
    gbm.neg_factored_gradient[-2.1] = grad;
    double temp = gbm.forest[-2.1][0].neg_variance_gain(more_idx,less_idx);
    EXPECT_EQ(-2.2,temp);//((1+1+1)^2/3 + (2+2)^2/2)/5 = (3+8)/5 = 2.2
    
    grad = vector<double>{1,1,-2,2,-2};
    more_idx = vector<int>{0,1,2};
    less_idx = vector<int>{3,4};
    
    gbm.neg_factored_gradient[-2.1] = grad;
    temp = gbm.forest[-2.1][0].neg_variance_gain(more_idx,less_idx);
    EXPECT_EQ(0,temp);//((1+1-2)^2/3 + (2-2)^2/2)/5 = 0
    
    grad = vector<double>{1, 1, 1.2, -2.4, 1, 1, 3.6, 1, 1, 0.6};
    more_idx = vector<int>{0,1,4,5,7,8};
    less_idx = vector<int>{2,3,6,9};
    
    gbm.neg_factored_gradient[-2.1] = grad;
    temp = gbm.forest[-2.1][0].neg_variance_gain(more_idx,less_idx);
    EXPECT_EQ(-0.825,temp);//((6)^2/6 + (3)^2/4)/10 = 0.825
}



TEST_F(LightGBMTest, loss_func_for_leaf){
    LightGBM_clf gbm;
    Eigen::MatrixXd y_train,temp;
    
    y_train = Eigen::MatrixXd(5,1);
    y_train << 1, 1, 2, 2, -2.1;
    temp = Eigen::MatrixXd(5,1);
    temp<<log(10),log(0.1),log(1),log(2),log(50);
    gbm.cash_result[1] = temp;
    temp<<log(5),log(0.05),log(1.5),log(3),log(50);
    gbm.cash_result[2] = temp;
    temp<<log(1),log(0.01),log(2.5),log(3),log(100);
    gbm.cash_result[-2.1] = temp;
    
    vector<int> data_idx{0};
    gbm.forest[1].resize(1);
    gbm.forest[1][0].y_train = &y_train;
    gbm.forest[1][0].set_forest(&gbm);
    double loss = gbm.forest[1][0].loss_func_for_leaf(data_idx,1,0);
    EXPECT_EQ(-log(10)+log(10 + 5 + 1),loss);
    loss = gbm.forest[1][0].loss_func_for_leaf(data_idx,1,log(3));
    EXPECT_EQ(round(1000*(-log(10) - log(3) + log(10 * 3 + 5 + 1))),round(1000*loss));
    
    gbm.forest[2].resize(1);
    gbm.forest[2][0].y_train = &y_train;
    gbm.forest[2][0].set_forest(&gbm);
    loss = gbm.forest[2][0].loss_func_for_leaf(data_idx,2,log(2));
    EXPECT_EQ(-log(5) + log(10 + 5*2 + 1),loss);
    
    data_idx = vector<int>{0,3};
    loss = gbm.forest[1][0].loss_func_for_leaf(data_idx,1,0);
    EXPECT_EQ(-log(10) + log(10 + 5 + 1) - log(2) + log(2 + 3 + 3),loss);
    loss = gbm.forest[1][0].loss_func_for_leaf(data_idx,1,log(5));
    EXPECT_EQ(-log(10) - log(5) + log(10 * 5 + 5 + 1) - log(2) + log(2 * 5 + 3  + 3),loss);
    
}

TEST_F(LightGBMTest, leaf_solver){
    LightGBM_clf gbm;
    Eigen::MatrixXd y_train,temp;
    
    y_train = Eigen::MatrixXd(4,1);
    y_train << 1, 2, 1, -2.1;
    temp = Eigen::MatrixXd(4,1);
    temp<<1,1,1,1;
    gbm.cash_result[1] = temp;
    temp<<1,1,1,1;
    gbm.cash_result[2] = temp;
    temp<<1,1,1,1;
    gbm.cash_result[-2.1] = temp;
    
    vector<int> data_idx{0};
    gbm.forest[1].resize(1);
    gbm.forest[1][0].y_train = &y_train;
    gbm.forest[1][0].set_forest(&gbm);
    double gamma = gbm.forest[1][0].leaf_solver(data_idx,1);
    EXPECT_EQ(1000 * 2, round(1000 * gamma));
    gamma = gbm.forest[1][0].leaf_solver(data_idx,2);
    EXPECT_EQ(1000 * -2, round(1000 * gamma));
    
    
    data_idx = vector<int>{0,1,2,3};
    gamma = gbm.forest[1][0].leaf_solver(data_idx,1);
    EXPECT_EQ(1000 * 0.7, round(1000 * gamma));
    
    
    temp<<1,1,1,1;
    gbm.cash_result[1] = temp;
    temp<<2,0.5,1.2,-3;
    gbm.cash_result[2] = temp;
    temp<<3,-2.1,1,-1;
    gbm.cash_result[-2.1] = temp;
    
    gamma = gbm.forest[1][0].leaf_solver(data_idx,1);
    EXPECT_EQ(1000 * 0.2, round(1000 * gamma));
    
    gamma = gbm.forest[1][0].leaf_solver(data_idx,2);
    EXPECT_EQ(1000 * 0.0, round(1000 * gamma));
    
    gamma = gbm.forest[1][0].leaf_solver(data_idx,-2.1);
    EXPECT_EQ(1000 * -0.2, round(1000 * gamma));
    
}

TEST_F(LightGBMTest, random_sampling_without_replacement){
    LightGBM_clf gbm;
    gbm.rand->mt_rand.seed(0);
    
    const vector<int> idx1 = gbm.rand->random_sampling_without_replacement(10,15,0);
    vector<int> ans1{0,1,4,6,8,9,10,11,12,13};
    
    for(int i = 0; i < idx1.size(); ++i){
        EXPECT_EQ(ans1[i], idx1[i]);
    }
    
    const vector<int> idx2 = gbm.rand->random_sampling_without_replacement(10,20,30);
    vector<int> ans2{31,33,34,38,39,41,44,45,46,48};

    for(int i = 0; i < idx2.size(); ++i){
        EXPECT_EQ(ans2[i], idx2[i]);
    }
}


TEST_F(LightGBMTest, GOS_sampling){
    
    LightGBM_clf gbm;
    gbm.rand->mt_rand.seed(0);
    map<double,vector<double>> f_grad;
    
    f_grad[1] = vector<double>{0,0,0,0,1,1,1,1,1,1};
    f_grad[2] = vector<double>{1,2,3,4,5,6,7,8,9,0};
    f_grad[-2.1] = vector<double>{0,-2,0,0,1,1,0,-1,-1,0};
    
    map<double,vector<int>>gos_result = gbm.rand->GOS_sampling(f_grad, 0.3, 0.3);
    map<double,vector<int>>gos_result2 = gbm.rand->GOS_sampling(f_grad, 0.1, 0.5);
    map<double,vector<int>>gos_result3 = gbm.rand->GOS_sampling(f_grad, 1.0, 0.0);
    vector<int> ans1{4,5,6,8,1,2};
    vector<int> ans2{8,7,6,5,3,1};
    vector<int> ans_m2_1{1,4,5,3,6,9};
    
    vector<int> ans1_2{4,7,8,9,1,2};
    vector<int> ans2_2{8,4,2,1,0,9};
    vector<int> ans_m2_1_2{1,4,5,0,2,6};
    
    vector<int> ans1_3{4,5,6,7,8,9,0,1,2,3};
    vector<int> ans2_3{8,7,6,5,4,3,2,1,0,9};
    vector<int> ans_m2_1_3{1,4,5,7,8,0,2,3,6,9};
    
    
    for(int i = 0;i < gos_result.cbegin()->second.size() ;++i){
        EXPECT_EQ(ans1[i], gos_result[1][i]);
        EXPECT_EQ(ans2[i], gos_result[2][i]);
        EXPECT_EQ(ans_m2_1[i], gos_result[-2.1][i]);
        
        EXPECT_EQ(ans1_2[i], gos_result2[1][i]);
        EXPECT_EQ(ans2_2[i], gos_result2[2][i]);
        EXPECT_EQ(ans_m2_1_2[i], gos_result2[-2.1][i]);
        
        EXPECT_EQ(ans1_3[i], gos_result3[1][i]);
        EXPECT_EQ(ans2_3[i], gos_result3[2][i]);
        EXPECT_EQ(ans_m2_1_3[i], gos_result3[-2.1][i]);
    }
}

 
TEST_F(LightGBMTest, fit){
    //変数宣言
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    vector<int> data_idx;
    
    LightGBM_clf gbm;
    LightGBM_clf::LightGBM_clf_param param;
    param.high_gradient_sampling_ratio = 0.5;
    param.low_gradient_sampling_ratio = 0.5;
    param.estimators_num = 2;
    param.tree_param.max_depth=2;
    param.random_seed = 100;
    
    gbm.set_param(param);
    
    X_train = Eigen::MatrixXd(8,2);
    X_train <<
    0,0,
    0,0,
    1,0,
    1,0,
    1,0,
    1,0,
    1,0,
    1,0;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 1,1,0,0,0,0,0,0;
    
    gbm.fit(X_train, y_train);
    
    EXPECT_EQ(2,gbm.forest.size());
    EXPECT_EQ(2,gbm.forest.begin()->second.size());
    
    //cout<<gbm.cash_result[1]<<endl;
    //cout<<gbm.cash_result[0]<<endl;
    
    EXPECT_EQ("/leaf_val->1.09861",gbm.forest[0][0].tree.get_leaf_list()[0]->path);
    auto list = gbm.forest[0][1].tree.get_leaf_list();
    EXPECT_EQ("/dep_1,f_idx_0,spt_1,more/leaf_val->2",list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_1,less/leaf_val->-2",list[1]->path);
    
    EXPECT_EQ("/leaf_val->0",gbm.forest[1][0].tree.get_leaf_list()[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_1,more/leaf_val->-2",gbm.forest[1][1].tree.get_leaf_list()[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_1,less/leaf_val->2",gbm.forest[1][1].tree.get_leaf_list()[1]->path);
    
 
    param.estimators_num = 5;
    gbm.set_param(param);
    
    y_train << 1,1,0,2,3,-2.1,0,0;
    gbm.fit(X_train, y_train);
    
    EXPECT_EQ(5,gbm.forest.size());
    EXPECT_EQ(5,gbm.forest.begin()->second.size());
 
}


TEST_F(LightGBMTest, predict){
    // 変数宣言
    Eigen::MatrixXd X_train;
    Eigen::MatrixXd y_train;
    vector<int> data_idx;
    LightGBM_clf gbm;
    LightGBM_clf::LightGBM_clf_param param;
    param.high_gradient_sampling_ratio = 0.5;
    param.low_gradient_sampling_ratio = 0.5;
    param.estimators_num = 3;
    param.random_seed = 0;
    param.tree_param.min_samples_split = 3;
    param.tree_param.max_depth = 2;
    
    gbm.set_param(param);
    
    X_train = Eigen::MatrixXd(6,2);
    X_train <<
    0,  0,
    1,  0,
    0,  0,
    1,  1,
    1,  1,
    0,  1;
    
    y_train = Eigen::MatrixXd(6,1);
    y_train << 0,2,0,1,1,1;
    
    gbm.fit(X_train,y_train);
    
    Eigen::MatrixXd X_test(3,2);
    X_test <<
    0,  1,
    1,  0,
    0,  0;
    
    Eigen::MatrixXd y_test(3,1);
    y_test << 1,2,0;
    
    Eigen::MatrixXd result = gbm.predict(X_test);

    EXPECT_EQ(y_test, result);
    
    
    X_train = Eigen::MatrixXd(8,7);
    X_train <<
    0,  1,  0,  7,  1,   0,   1,
    0,  1,  0,  5,  1, 1.9,   1,
    0,  1,  0,  6,  1,-2.6,   1,
    0,  1,  0,  3,  1, 0.8,   1,
    0,  1,  0,  3,  1, 1.2,   1,
    0,  1,  0,  3,  1,-1.3,   1,
    0,  1,  0,  4,  1, 2.0,   1,
    0,  1,  0,  2,  1, 4.1,   1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 3,3,3,3,1,1,2,1;
    
    
    param.estimators_num = 10;
    param.random_seed = 0;
    param.tree_param.min_samples_split = 2;
    param.tree_param.max_depth = 3;
    
    gbm.set_param(param);
    
    gbm.fit(X_train,y_train);
    
    X_test = Eigen::MatrixXd(12,7);
    
    X_test <<
    0,  0,  0,  7,   1,   0,   1,
    0,  0,  0,  9,   12,   -356,   4.56,
    0,  0,  0,  5,   1, 1.9,   1,
    0,  0,  0,  5.75,   -367, 1.989,   -1,
    0,  0,  0,  4,   1, 2.0,   1,
    0,  0,  0,  4.999,   1, 2.2,   1,
    0,  0,  0,  3,   1, 1.2,   1,
    0,  0,  0,  2.9,   1, 200,   1,
    0,  0,  0,  3,   1, 0.8,   1,
    0,  0,  0,  3.999,   -1, 1.19,   1,
    0,  0,  0,  3,   1,-1.3,   1,
    0,  0,  0,  3.999,   1234, 0.7999,   -99999999;
    
    y_test = Eigen::MatrixXd(12,1);
    y_test <<  3,3,3,3,2,2,1,1,3,3,1,1;
    
    result = gbm.predict(X_test);
    
    for(int i = 0; i < y_test.cols(); ++i){
            EXPECT_EQ(y_test(i,0), result(i,0));
    }
 
}


TEST_F(LightGBMTest, non_0_count){
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(6,4);
    X_train <<
    0,  0,  0,  0.001,
    1,  0,  0,  0.001,
    0,  0,  0,  0.001,
    1,  1,  0,  0.001,
    1,  1,  0,  0.001,
    0,  1,  0,  0.001;
    
    vector<int> non_zero = gbm.exclusive_feature_bundling.non_0_count(X_train);
    vector<int> ans{3,3,0,6};
    
    for(int i = 0; i < non_zero.size(); ++i){
        EXPECT_EQ(non_zero[i], ans[i]);
    }
}


TEST_F(LightGBMTest, conflict_count){
    
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(6,4);
    X_train <<
    0,  0,  0,  0.001,
    1,  0,  0,  0.001,
    0,  0,  0,  0.001,
    1,  1.3,  0,  0.001,
    1,  -15.6,  0,  0.001,
    0,  -1.125,  0,  0.001;
    
    int result = gbm.exclusive_feature_bundling.conflict_count(X_train.col(0), X_train.col(1));
    EXPECT_EQ(2, result);
    
    result = gbm.exclusive_feature_bundling.conflict_count(X_train.col(0), X_train.col(2));
    EXPECT_EQ(0, result);
    
    result = gbm.exclusive_feature_bundling.conflict_count(X_train.col(0), X_train.col(3));
    EXPECT_EQ(3, result);
    
    result = gbm.exclusive_feature_bundling.conflict_count(X_train.col(1), X_train.col(3));
    EXPECT_EQ(3, result);
    
}


TEST_F(LightGBMTest, greedy_bundling){
    
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(6,5);
    X_train <<
    0,  0,      0,  1,  -0.1,
    1,  0,      0,  1,  0,
    0,  0,      0,  1,  -0.1,
    1,  1.3,    0,  1,  0,
    1,  -15.6,  0,  1,  0,
    0,  -1.125, 0,  1,  -0.1;
    
    vector<vector<int>> bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,0);
    
    EXPECT_EQ(3, bundles[0][0]);
    EXPECT_EQ(2, bundles[0][1]);
    EXPECT_EQ(4, bundles[1][0]);
    EXPECT_EQ(0, bundles[1][1]);
    EXPECT_EQ(1, bundles[2][0]);
    
    X_train = Eigen::MatrixXd(6,6);
    X_train <<
    1,  2,  1.3,    1,  1,  -0.1,
    0,  2,  -15.6,  2,  1,  -0.1,
    0,  0,  0,      3,  1,  -0.1,
    0,  0,  0,      4,  1,  -0.1,
    0,  0,  0,      0,  1,  -0.1,
    0,  0,  0,      0,  0,  -0.1;
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,3);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(2, bundles[0][1]);
    EXPECT_EQ(4, bundles[1][0]);
    EXPECT_EQ(1, bundles[1][1]);
    EXPECT_EQ(3, bundles[2][0]);
    EXPECT_EQ(0, bundles[2][1]);
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,4);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(3, bundles[0][1]);
    EXPECT_EQ(4, bundles[1][0]);
    EXPECT_EQ(2, bundles[1][1]);
    EXPECT_EQ(0, bundles[1][2]);
    EXPECT_EQ(1, bundles[2][0]);
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,0);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[1][0]);
    EXPECT_EQ(3, bundles[2][0]);
    EXPECT_EQ(2, bundles[3][0]);
    EXPECT_EQ(1, bundles[4][0]);
    EXPECT_EQ(0, bundles[5][0]);
    
    X_train <<
    1,  0,  0,  1,  0,   0.0,
    0,  2,  0,  0,  1,   0.0,
    0,  0,  1,  0,  0,   0.1,
    1,  0,  0,  1,  0,   0.0,
    0,  1,  0,  0,  1,   0.0,
    0,  0,  1,  0,  0,   0.1;
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,0);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[1][0]);
    EXPECT_EQ(1, bundles[1][1]);
    EXPECT_EQ(0, bundles[1][2]);
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,1);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[1][0]);
    EXPECT_EQ(1, bundles[1][1]);
    EXPECT_EQ(0, bundles[1][2]);
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,2);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[0][3]);
    EXPECT_EQ(1, bundles[1][0]);
    EXPECT_EQ(0, bundles[1][1]);
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,6);
    
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[0][3]);
    EXPECT_EQ(1, bundles[0][4]);
    EXPECT_EQ(0, bundles[0][5]);
    
};


TEST_F(LightGBMTest, bundle_map_builder){
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(6,6);
    
    X_train <<
    1,  0,  0,  1,  0,      0.0,
    0,  2,  0,  0,  -2.3,   0.0,
    0,  0,  1,  0,  0,      0.2,
    1,  0,  0,  6,  0,      0.0,
    0,  1,  0,  0,  0.01,   0.0,
    0,  0,  1,  0,  0,      0.1;
    
    vector<vector<int>> bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,0);
    
    map<int, map<double,int>>result_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundles[0]);
    
    EXPECT_EQ(5, result_map[3][1]);
    EXPECT_EQ(6, result_map[3][6]);
    EXPECT_EQ(3, result_map[4][-2.3]);
    EXPECT_EQ(4, result_map[4][0.01]);
    EXPECT_EQ(2, result_map[5][0.1]);
    EXPECT_EQ(1, result_map[5][0.2]);
    
    result_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundles[1]);
    
    EXPECT_EQ(4, result_map[0][1]);
    EXPECT_EQ(3, result_map[1][1]);
    EXPECT_EQ(2, result_map[1][2]);
    EXPECT_EQ(1, result_map[2][1]);
    
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,2);
    result_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundles[0]);
    
    EXPECT_EQ(7, result_map[2][1]);
    EXPECT_EQ(5, result_map[3][1]);
    EXPECT_EQ(6, result_map[3][6]);
    EXPECT_EQ(3, result_map[4][-2.3]);
    EXPECT_EQ(4, result_map[4][0.01]);
    EXPECT_EQ(2, result_map[5][0.1]);
    EXPECT_EQ(1, result_map[5][0.2]);
    
    result_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundles[1]);
    
    EXPECT_EQ(3, result_map[0][1]);
    EXPECT_EQ(2, result_map[1][1]);
    EXPECT_EQ(1, result_map[1][2]);
    
}


TEST_F(LightGBMTest, mapping){
    
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(7,6);
    
    X_train <<
    1,  0,  0,  1,  0,      0.0,
    0,  2,  0,  0,  -2.3,   0.0,
    0,  0,  1,  0,  0,      0.2,
    1,  0,  0,  6,  0,      0.0,
    0,  1,  0,  0,  0.01,   0.0,
    0,  0,  1,  0,  0,      0.1,
    0,  0,  0,  0,  0,      0.0;
    
    vector<vector<int>> bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,0);
    vector<int> bundle = bundles[0];
    map<int, map<double,int>>bundle_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundle);
    Eigen::MatrixXd X_m = gbm.exclusive_feature_bundling.mapping(X_train,bundle,bundle_map);
    
    EXPECT_EQ(5, X_m(0,0));
    EXPECT_EQ(3, X_m(1,0));
    EXPECT_EQ(1, X_m(2,0));
    EXPECT_EQ(6, X_m(3,0));
    EXPECT_EQ(4, X_m(4,0));
    EXPECT_EQ(2, X_m(5,0));
    EXPECT_EQ(0, X_m(6,0));

    bundle = bundles[1];
    bundle_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundle);
    X_m = gbm.exclusive_feature_bundling.mapping(X_train,bundle,bundle_map);
    
    EXPECT_EQ(4, X_m(0,0));
    EXPECT_EQ(2, X_m(1,0));
    EXPECT_EQ(1, X_m(2,0));
    EXPECT_EQ(4, X_m(3,0));
    EXPECT_EQ(3, X_m(4,0));
    EXPECT_EQ(1, X_m(5,0));
    EXPECT_EQ(0, X_m(6,0));
    
    bundles = gbm.exclusive_feature_bundling.greedy_bundling(X_train,2);
    bundle = bundles[0];
    bundle_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundle);
    X_m = gbm.exclusive_feature_bundling.mapping(X_train,bundle,bundle_map);
    
    EXPECT_EQ(5, X_m(0,0));
    EXPECT_EQ(3, X_m(1,0));
    EXPECT_EQ(1, X_m(2,0));
    EXPECT_EQ(6, X_m(3,0));
    EXPECT_EQ(4, X_m(4,0));
    EXPECT_EQ(2, X_m(5,0));
    EXPECT_EQ(0, X_m(6,0));
    
    bundle = bundles[1];
    bundle_map = gbm.exclusive_feature_bundling.bundle_map_builder(X_train, bundle);
    X_m = gbm.exclusive_feature_bundling.mapping(X_train,bundle,bundle_map);
    
    EXPECT_EQ(3, X_m(0,0));
    EXPECT_EQ(1, X_m(1,0));
    EXPECT_EQ(0, X_m(2,0));
    EXPECT_EQ(3, X_m(3,0));
    EXPECT_EQ(2, X_m(4,0));
    EXPECT_EQ(0, X_m(5,0));
    EXPECT_EQ(0, X_m(6,0));
    
}


TEST_F(LightGBMTest, EFB_fit){
    
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train;
    X_train = Eigen::MatrixXd(7,6);
    
    X_train <<
    1,  0,  0,  1,  0,      0.0,
    0,  2,  0,  0,  -2.3,   0.0,
    0,  0,  1,  0,  0,      0.2,
    1,  0,  0,  6,  0,      0.0,
    0,  1,  0,  0,  0.01,   0.0,
    0,  0,  1,  0,  0,      0.1,
    0,  0,  0,  0,  0,      0.0;
    
    gbm.exclusive_feature_bundling.fit(X_train, 0);
    
    auto& bundles = gbm.exclusive_feature_bundling.bundles;
    auto& bundles_map = gbm.exclusive_feature_bundling.bundles_map;
    
    EXPECT_EQ(2, bundles.size());
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[1][0]);
    EXPECT_EQ(1, bundles[1][1]);
    EXPECT_EQ(0, bundles[1][2]);
    
    EXPECT_EQ(2, bundles_map.size());
    EXPECT_EQ(3, bundles_map[0].size());
    EXPECT_EQ(1, bundles_map[0][5][0.2]);
    EXPECT_EQ(2, bundles_map[0][5][0.1]);
    EXPECT_EQ(3, bundles_map[0][4][-2.3]);
    EXPECT_EQ(4, bundles_map[0][4][0.01]);
    EXPECT_EQ(5, bundles_map[0][3][1]);
    EXPECT_EQ(6, bundles_map[0][3][6]);
    
    EXPECT_EQ(3, bundles_map[1].size());
    EXPECT_EQ(1, bundles_map[1][2][1]);
    EXPECT_EQ(2, bundles_map[1][1][2]);
    EXPECT_EQ(3, bundles_map[1][1][1]);
    EXPECT_EQ(4, bundles_map[1][0][1]);

    
    gbm.exclusive_feature_bundling.fit(X_train, 2);
    
    bundles = gbm.exclusive_feature_bundling.bundles;
    bundles_map = gbm.exclusive_feature_bundling.bundles_map;
    
    EXPECT_EQ(2, bundles.size());
    EXPECT_EQ(5, bundles[0][0]);
    EXPECT_EQ(4, bundles[0][1]);
    EXPECT_EQ(3, bundles[0][2]);
    EXPECT_EQ(2, bundles[0][3]);
    EXPECT_EQ(1, bundles[1][0]);
    EXPECT_EQ(0, bundles[1][1]);
    
    EXPECT_EQ(2, bundles_map.size());
    EXPECT_EQ(4, bundles_map[0].size());
    EXPECT_EQ(1, bundles_map[0][5][0.2]);
    EXPECT_EQ(2, bundles_map[0][5][0.1]);
    EXPECT_EQ(3, bundles_map[0][4][-2.3]);
    EXPECT_EQ(4, bundles_map[0][4][0.01]);
    EXPECT_EQ(5, bundles_map[0][3][1]);
    EXPECT_EQ(6, bundles_map[0][3][6]);
    EXPECT_EQ(7, bundles_map[0][2][1]);
    
    EXPECT_EQ(2, bundles_map[1].size());
    EXPECT_EQ(1, bundles_map[1][1][2]);
    EXPECT_EQ(2, bundles_map[1][1][1]);
    EXPECT_EQ(3, bundles_map[1][0][1]);
    
}


TEST_F(LightGBMTest, transfrom){
    
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train, X_result, X_train_2;
    X_train = Eigen::MatrixXd(7,6);
    
    X_train <<
    1,  0,  0,  1,  0,      0.0,
    0,  2,  0,  0,  -2.3,   0.0,
    0,  0,  1,  0,  0,      0.2,
    1,  0,  0,  6,  0,      0.0,
    0,  1,  0,  0,  0.01,   0.0,
    0,  0,  1,  0,  0,      0.1,
    0,  0,  0,  0,  0,      0.0;
    
    X_train_2 = Eigen::MatrixXd(4,6);
    X_train_2 <<
    1,  0,  1,  1,  0,      0.0,
    0,  2,  0,  0,  0,      0.0,
    1,  1,  0,  0,  -2.3,   0.1,
    0,  0,  0,  6,  0,      0.0;

    gbm.exclusive_feature_bundling.fit(X_train, 0);
    X_result = gbm.exclusive_feature_bundling.transform(X_train);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(3, X_result(1,0));
    EXPECT_EQ(1, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    EXPECT_EQ(4, X_result(4,0));
    EXPECT_EQ(2, X_result(5,0));
    EXPECT_EQ(0, X_result(6,0));
    
    EXPECT_EQ(4, X_result(0,1));
    EXPECT_EQ(2, X_result(1,1));
    EXPECT_EQ(1, X_result(2,1));
    EXPECT_EQ(4, X_result(3,1));
    EXPECT_EQ(3, X_result(4,1));
    EXPECT_EQ(1, X_result(5,1));
    EXPECT_EQ(0, X_result(6,1));
    
    X_result = gbm.exclusive_feature_bundling.transform(X_train_2);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(0, X_result(1,0));
    EXPECT_EQ(2, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    
    EXPECT_EQ(1, X_result(0,1));
    EXPECT_EQ(2, X_result(1,1));
    EXPECT_EQ(3, X_result(2,1));
    EXPECT_EQ(0, X_result(3,1));
    
    
    gbm.exclusive_feature_bundling.fit(X_train, 2);
    X_result = gbm.exclusive_feature_bundling.transform(X_train);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(3, X_result(1,0));
    EXPECT_EQ(1, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    EXPECT_EQ(4, X_result(4,0));
    EXPECT_EQ(2, X_result(5,0));
    EXPECT_EQ(0, X_result(6,0));
    
    EXPECT_EQ(3, X_result(0,1));
    EXPECT_EQ(1, X_result(1,1));
    EXPECT_EQ(0, X_result(2,1));
    EXPECT_EQ(3, X_result(3,1));
    EXPECT_EQ(2, X_result(4,1));
    EXPECT_EQ(0, X_result(5,1));
    EXPECT_EQ(0, X_result(6,1));
    
    X_result = gbm.exclusive_feature_bundling.transform(X_train_2);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(0, X_result(1,0));
    EXPECT_EQ(2, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    
    EXPECT_EQ(3, X_result(0,1));
    EXPECT_EQ(1, X_result(1,1));
    EXPECT_EQ(2, X_result(2,1));
    EXPECT_EQ(0, X_result(3,1));
    
}

TEST_F(LightGBMTest, fit_transfrom){
    LightGBM_clf gbm;
    Eigen::MatrixXd X_train, X_result;
    X_train = Eigen::MatrixXd(7,6);
    
    X_train <<
    1,  0,  0,  1,  0,      0.0,
    0,  2,  0,  0,  -2.3,   0.0,
    0,  0,  1,  0,  0,      0.2,
    1,  0,  0,  6,  0,      0.0,
    0,  1,  0,  0,  0.01,   0.0,
    0,  0,  1,  0,  0,      0.1,
    0,  0,  0,  0,  0,      0.0;
    
    X_result = gbm.exclusive_feature_bundling.fit_transform(X_train);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(3, X_result(1,0));
    EXPECT_EQ(1, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    EXPECT_EQ(4, X_result(4,0));
    EXPECT_EQ(2, X_result(5,0));
    EXPECT_EQ(0, X_result(6,0));
    
    EXPECT_EQ(4, X_result(0,1));
    EXPECT_EQ(2, X_result(1,1));
    EXPECT_EQ(1, X_result(2,1));
    EXPECT_EQ(4, X_result(3,1));
    EXPECT_EQ(3, X_result(4,1));
    EXPECT_EQ(1, X_result(5,1));
    EXPECT_EQ(0, X_result(6,1));
    
    
    X_result = gbm.exclusive_feature_bundling.fit_transform(X_train, 2);
    
    EXPECT_EQ(5, X_result(0,0));
    EXPECT_EQ(3, X_result(1,0));
    EXPECT_EQ(1, X_result(2,0));
    EXPECT_EQ(6, X_result(3,0));
    EXPECT_EQ(4, X_result(4,0));
    EXPECT_EQ(2, X_result(5,0));
    EXPECT_EQ(0, X_result(6,0));
    
    EXPECT_EQ(3, X_result(0,1));
    EXPECT_EQ(1, X_result(1,1));
    EXPECT_EQ(0, X_result(2,1));
    EXPECT_EQ(3, X_result(3,1));
    EXPECT_EQ(2, X_result(4,1));
    EXPECT_EQ(0, X_result(5,1));
    EXPECT_EQ(0, X_result(6,1));
    
}



#endif /* light_gbm_test_h */
