//
//  clf_tree_test.h
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/13.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef clf_tree_test_h
#define clf_tree_test_h

#include "gtest/gtest.h"
#include "clf_tree.hpp"

namespace{
    class ClfTreeTest : public ::testing::Test{
    protected:
        ClfTreeTest(){}
        virtual ~ClfTreeTest(){}
        virtual void SetUp(){}
        virtual void TearDown(){}
    };
}


TEST_F(ClfTreeTest, max_depth)
{
    
    CLF_tree::CLF_tree_param param;
    param.max_depth = 3;
    
    CLF_tree clf_tree(param);
    
    // depth 0->1
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    // depth 1->2
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    EXPECT_EQ(false, clf_tree.is_max_depth());
    
    // depth 2->3
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    EXPECT_EQ(true, clf_tree.is_max_depth());
    
}


TEST_F(ClfTreeTest, min_samples_split)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 3;
    
    CLF_tree clf_tree(param);
    
    vector<int> idx;
    for(int i = 0;i < param.min_samples_split;i++)
        idx.push_back(i);
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    // idx.size() => param.min_samples_split
    EXPECT_EQ(false, clf_tree.is_min_samples_split());
    
    idx.pop_back();
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    // idx.size() < param.min_samples_split
    EXPECT_EQ(true, clf_tree.is_min_samples_split());
    
}


TEST_F(ClfTreeTest, stop_condition)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 3;
    param.max_depth = 3;
    
    CLF_tree clf_tree(param);
    
    // idx.size() -> param.min_samples_split then param.min_samples_split -> false
    vector<int> idx;
    for(int i = 0;i < param.min_samples_split;i++)
        idx.push_back(i);
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    // depth 0->1
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    // depth 1->2
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    
    EXPECT_EQ(false, clf_tree.stop_condition());
    
    // idx.size()<min_samples_split then is_min_samples_split() -> true
    idx.pop_back();
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    EXPECT_EQ(true, clf_tree.stop_condition());
    
    //is_min_samples_split() -> false
    idx.push_back(1);
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    // depth 2 -> 3 then is_max_depth() -> true
    clf_tree.tree.add_more_node();
    clf_tree.tree.to_more_node();
    
    EXPECT_EQ(true, clf_tree.stop_condition());
    
    // is_min_samples_split() -> false
    idx.pop_back();
    clf_tree.idx_size_for_is_min_samples_split = idx.size();
    
    // both true
    EXPECT_EQ(true, clf_tree.stop_condition());
    
}


TEST_F(ClfTreeTest, gini)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 3;
    param.max_depth = 3;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd y_train(10,1);
    y_train << 1,1,1,1,1,0,0,0,0,0;
    
    vector<int> more_idx{0,1,2,3,4};
    vector<int> less_idx{5,6,7,8,9};
    
    clf_tree.y_train = &y_train;
    
    EXPECT_EQ(0, clf_tree.gini(more_idx,less_idx));
    
    y_train << 1,0,1,0,1,0,1,0,1,0;
    
    //vector<int> more_idx{0,1,2,3,4};
    //vector<int> less_idx{5,6,7,8,9};
    
    clf_tree.y_train = &y_train;
    
    // (1 - (2/5)^2 -(3/5)^2)*5/10 + (1 - (2/5)^2 -(3/5)^2)*5/10
    // = 12/25 = 0.48
    EXPECT_EQ(0.48, clf_tree.gini(more_idx,less_idx));
    
    
    y_train << 1,1,0,0,0,0,0,0,1,1;
    clf_tree.y_train = &y_train;
    more_idx = vector<int>{0,1,2,3,4,5,6,7};
    less_idx = vector<int>{8,9};
    
    // (1 - (2/8)^2 - (6/8)^2)*8/10 + (1 - (2/2)^2)*2/10
    // =(1 - 1/16 - 9/16)*4/5 + 0
    // = 6/16*4/5 = 3/10 = 0.3
    EXPECT_EQ(0.3, clf_tree.gini(more_idx,less_idx));
    
    
    
    y_train << 0,0,0,0,0,0,0,0,0,0;
    clf_tree.y_train = &y_train;
    more_idx = vector<int>{0,1,2,3,4,5,6,7,8,9};
    less_idx = vector<int>{};
    
    // (1 - (10/10)^2)*10/10 + (1 - (0)^2)*0/10
    EXPECT_EQ(0, clf_tree.gini(more_idx,less_idx));
    
}


TEST_F(ClfTreeTest, make_split)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 3;
    param.max_depth = 3;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd X_train(5,3);
    X_train << 1,1,1,
               1,1,0,
               0,1,0,
               0,0,1,
               1,0,1;
    
    Eigen::MatrixXd y_train(5,1);
    y_train << 1,1,1,0,0;
    
    vector<int> data_idx{0,1,2,3,4};
    
    clf_tree.X_train = &X_train;
    clf_tree.y_train = &y_train;
    CLF_tree::Tree_Split split = clf_tree.make_split(data_idx);
    
    EXPECT_EQ(1, split.feature_idx);
    EXPECT_EQ(3, split.more_idx.size());
    EXPECT_EQ(2, split.less_idx.size());
    EXPECT_EQ(1, split.split_value);
    
    X_train = Eigen::MatrixXd(8,3);
    X_train << 1,0,1,
               1,0,1,
               1,0,1,
               0,0,1,
               0,0,1,
               0,0,1,
               0,1,1,
               0,1,1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 1,1,1,1,0,0,0,0;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    clf_tree.X_train = &X_train;
    clf_tree.y_train = &y_train;
    split = clf_tree.make_split(data_idx);
    
    EXPECT_EQ(0, split.feature_idx);
    EXPECT_EQ(3, split.more_idx.size());
    EXPECT_EQ(5, split.less_idx.size());
    EXPECT_EQ(1, split.split_value);
    
    data_idx = vector<int>{0,1,2,3,6,7};
    
    split = clf_tree.make_split(data_idx);
    
    EXPECT_EQ(1, split.feature_idx);
    EXPECT_EQ(2, split.more_idx.size());
    EXPECT_EQ(4, split.less_idx.size());
    EXPECT_EQ(1, split.split_value);

    
    X_train << 7,   0,   1,
               5, 1.9,   1,
               6,-2.6,   1,
               3, 0.8,   1,
               3, 1.2,   1,
               3,-1.3,   1,
               4, 2.0,   1,
               2, 4.1,   1;
    
    y_train << 1,1,1,1,0,0,0,0;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    clf_tree.X_train = &X_train;
    clf_tree.y_train = &y_train;
    split = clf_tree.make_split(data_idx);
    
    EXPECT_EQ(0, split.feature_idx);
    EXPECT_EQ(3, split.more_idx.size());
    EXPECT_EQ(5, split.less_idx.size());
    EXPECT_EQ(5, split.split_value);
    
    data_idx = vector<int>{0,1,2,3,6,7};
    
    split = clf_tree.make_split(data_idx);
    
    EXPECT_EQ(1, split.feature_idx);
    EXPECT_EQ(2, split.more_idx.size());
    EXPECT_EQ(4, split.less_idx.size());
    EXPECT_EQ(2, split.split_value);


}

TEST_F(ClfTreeTest, rc_growth_tree)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 2;
    param.max_depth = 4;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd X_train(8,3);
    X_train << 1,1,1,
               1,1,0,
               1,1,0,
               1,1,0,
               1,1,0,
               0,1,0,
               0,0,1,
               0,0,1;
    
    Eigen::MatrixXd y_train(8,1);
    y_train << 1,1,1,1,1,1,0,0;
    
    vector<int> data_idx{0,1,2,3,4,5,6,7};
    
    clf_tree.X_train = &X_train;
    clf_tree.y_train = &y_train;
    clf_tree.rc_growth_tree(data_idx);
    auto leaf_list = clf_tree.tree.get_leaf_list();
    
    /*
    for(auto iter = leaf_list.cbegin(); iter != leaf_list.cend(); iter++){
        cout<<(*iter)->path<<endl;
    }
    */
    
    EXPECT_EQ("/dep_1,f_idx_1,spt_1,more/dep_2,f_idx_0,spt_1,more/leaf_val->1",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_1,spt_1,more/dep_2,f_idx_0,spt_1,less/leaf_val->1",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_1,spt_1,less/leaf_val->0",leaf_list[2]->path);
    
    X_train = Eigen::MatrixXd(8,4);
    X_train << 7,   1,   0, 1,
               5,   1, 1.9, 1,
               6,   1,-2.6, 1,
               3,   1, 0.8, 1,
               3,   1, 1.2, 1,
               3,   1,-1.3, 1,
               4,   1, 2.0, 1,
               2,   1, 4.1, 1;
    
    y_train = Eigen::MatrixXd(8,1);
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    clf_tree.X_train = &X_train;
    clf_tree.y_train = &y_train;
    clf_tree.tree.del_all_nodes();
    clf_tree.tree.set_root_node();
    clf_tree.rc_growth_tree(data_idx);
    leaf_list = clf_tree.tree.get_leaf_list();
    
    /*
    for(auto iter = leaf_list.cbegin(); iter != leaf_list.cend(); iter++){
        cout<<(*iter)->path<<endl;
    }
    */
    
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,more/leaf_val->3",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,less/leaf_val->3",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,more/leaf_val->-2",leaf_list[2]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,more/dep_4,f_idx_0,spt_3,more/leaf_val->-1",leaf_list[3]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,more/dep_4,f_idx_0,spt_3,less/leaf_val->-1",leaf_list[4]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,less/dep_4,f_idx_2,spt_0.8,more/leaf_val->3",leaf_list[5]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,less/dep_4,f_idx_2,spt_0.8,less/leaf_val->-1",leaf_list[6]->path);
    
    
    // min_samples_splitの変更
    param.min_samples_split = 3;
    param.max_depth = 4;
    clf_tree.set_param(param);
    
    X_train << 7,   1,   0,   1,
               5,   1, 1.9,   1,
               6,   1,-2.6,   1,
               3,   1, 0.8,   1,
               3,   1, 1.2,   1,
               3,   1,-1.3,   1,
               4,   1, 2.0,   1,
               2,   1, 4.1,   1;
    
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    clf_tree.tree.del_all_nodes();
    clf_tree.tree.set_root_node();
    clf_tree.rc_growth_tree(data_idx);
    leaf_list = clf_tree.tree.get_leaf_list();
    
    /*
     for(auto iter = leaf_list.cbegin(); iter != leaf_list.cend(); iter++){
     cout<<(*iter)->path<<endl;
     }
     */
    
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,more/leaf_val->3",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,less/leaf_val->3",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,more/leaf_val->-2",leaf_list[2]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,more/leaf_val->-1",leaf_list[3]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,less/leaf_val->-1",leaf_list[4]->path);
   
    // max_depthの変更
    param.min_samples_split = 3;
    param.max_depth = 2;
    clf_tree.set_param(param);
    
    X_train << 7,   1,   0,   1,
               5,   1, 1.9,   1,
               6,   1,-2.6,   1,
               3,   1, 0.8,   1,
               3,   1, 1.2,   1,
               3,   1,-1.3,   1,
               4,   1, 2.0,   1,
               2,   1, 4.1,   1;
    
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    data_idx = vector<int>{0,1,2,3,4,5,6,7};
    
    clf_tree.tree.del_all_nodes();
    clf_tree.tree.set_root_node();
    clf_tree.rc_growth_tree(data_idx);
    leaf_list = clf_tree.tree.get_leaf_list();
    
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,more/leaf_val->3",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,less/leaf_val->3",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,more/leaf_val->-2",leaf_list[2]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/leaf_val->-1",leaf_list[3]->path);
    
}

TEST_F(ClfTreeTest, fit)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 3;
    param.max_depth = 2;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd X_train(8,4);
    X_train << 7,   1,   0,   1,
               5,   1, 1.9,   1,
               6,   1,-2.6,   1,
               3,   1, 0.8,   1,
               3,   1, 1.2,   1,
               3,   1,-1.3,   1,
               4,   1, 2.0,   1,
               2,   1, 4.1,   1;
    
    Eigen::MatrixXd y_train(8,1);
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    clf_tree.fit(X_train,y_train);
    vector<Node*>leaf_list = clf_tree.tree.get_leaf_list();
    
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,more/leaf_val->3",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,less/leaf_val->3",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,more/leaf_val->-2",leaf_list[2]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/leaf_val->-1",leaf_list[3]->path);
    
    
    X_train << 7,   1,   0,   1,
               5,   1, 1.9,   1,
               6,   1,-2.6,   1,
               3,   1, 0.8,   1,
               3,   1, 1.2,   1,
               3,   1,-1.3,   1,
               4,   1, 2.0,   1,
               2,   1, 4.1,   1;
    
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    param.min_samples_split = 2;
    param.max_depth = 4;
    clf_tree.set_param(param);
    clf_tree.fit(X_train,y_train);
    leaf_list = clf_tree.tree.get_leaf_list();
    
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,more/leaf_val->3",leaf_list[0]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,more/dep_2,f_idx_0,spt_7,less/leaf_val->3",leaf_list[1]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,more/leaf_val->-2",leaf_list[2]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,more/dep_4,f_idx_0,spt_3,more/leaf_val->-1",leaf_list[3]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,more/dep_4,f_idx_0,spt_3,less/leaf_val->-1",leaf_list[4]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,less/dep_4,f_idx_2,spt_0.8,more/leaf_val->3",leaf_list[5]->path);
    EXPECT_EQ("/dep_1,f_idx_0,spt_5,less/dep_2,f_idx_0,spt_4,less/dep_3,f_idx_2,spt_1.2,less/dep_4,f_idx_2,spt_0.8,less/leaf_val->-1",leaf_list[6]->path);
    
}

TEST_F(ClfTreeTest, one_pred)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 2;
    param.max_depth = 4;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd X_train(8,4);
    X_train << 7,   1,   0,   1,
               5,   1, 1.9,   1,
               6,   1,-2.6,   1,
               3,   1, 0.8,   1,
               3,   1, 1.2,   1,
               3,   1,-1.3,   1,
               4,   1, 2.0,   1,
               2,   1, 4.1,   1;
    
    Eigen::MatrixXd y_train(8,1);
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    clf_tree.fit(X_train,y_train);
    
    Eigen::MatrixXd X_row(1,4);
    
    X_row << 7,   1,   0,   1;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));
    X_row << 9,   12,   -356,   4.56;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));

    X_row << 5,   1, 1.9,   1;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));
    X_row << 5.75,   -367, 1.989,   -1;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));
    
    X_row << 4,   1, 2.0,   1;
    EXPECT_EQ(-2, clf_tree.one_pred(X_row));
    X_row << 4.999,   1, 2.2,   1;
    EXPECT_EQ(-2, clf_tree.one_pred(X_row));
    
    X_row << 3,   1, 1.2,   1;
    EXPECT_EQ(-1, clf_tree.one_pred(X_row));
    X_row << 2.9,   1, 200,   1;
    EXPECT_EQ(-1, clf_tree.one_pred(X_row));
    
    X_row << 3,   1, 0.8,   1;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));
    X_row << 3.999,   -1, 1.19,   1;
    EXPECT_EQ(3, clf_tree.one_pred(X_row));
    
    X_row << 3,   1,-1.3,   1;
    EXPECT_EQ(-1, clf_tree.one_pred(X_row));
    X_row << 3.999,   1234, 0.7999,   -99999999;
    EXPECT_EQ(-1, clf_tree.one_pred(X_row));
    
}

TEST_F(ClfTreeTest, predict)
{
    CLF_tree::CLF_tree_param param;
    param.min_samples_split = 2;
    param.max_depth = 4;
    
    CLF_tree clf_tree(param);
    
    Eigen::MatrixXd X_train(8,4);
    X_train << 7,   1,   0,   1,
    5,   1, 1.9,   1,
    6,   1,-2.6,   1,
    3,   1, 0.8,   1,
    3,   1, 1.2,   1,
    3,   1,-1.3,   1,
    4,   1, 2.0,   1,
    2,   1, 4.1,   1;
    
    Eigen::MatrixXd y_train(8,1);
    y_train << 3,3,3,3,-1,-1,-2,-1;
    
    clf_tree.fit(X_train,y_train);
    
    Eigen::MatrixXd X_test(12,4);
    X_test << 7,   1,   0,   1,
              9,   12,   -356,   4.56,
              5,   1, 1.9,   1,
              5.75,   -367, 1.989,   -1,
              4,   1, 2.0,   1,
              4.999,   1, 2.2,   1,
              3,   1, 1.2,   1,
              2.9,   1, 200,   1,
              3,   1, 0.8,   1,
              3.999,   -1, 1.19,   1,
              3,   1,-1.3,   1,
              3.999,   1234, 0.7999,   -99999999;
    
    Eigen::MatrixXd y_test(12,1);
    y_test << 3,3,3,3,-2,-2,-1,-1,3,3,-1,-1;
    
    EXPECT_EQ(y_test, clf_tree.predict(X_test));
    
    
}
#endif /* clf_tree_test_h */
