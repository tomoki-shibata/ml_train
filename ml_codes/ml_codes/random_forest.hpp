//
//  random_forest.hpp
//  ml_codes
//
//  Created by 柴田 智喜 on 2018/01/29.
//  Copyright © 2018年 柴田 智喜. All rights reserved.
//

#ifndef random_forest_hpp
#define random_forest_hpp

#include "clf_tree.hpp"
#include <random>


class Random_Forest_clf{

// クラス内クラスの定義
public:
    // パラメータ
    struct Random_Forest_clf_param{
        CLF_tree::CLF_tree_param tree_param;
        int estimators_num;
        int feature_sample_num;
        int random_seed;
        Random_Forest_clf_param();
        Random_Forest_clf_param(Random_Forest_clf_param& param_inp);
    };
    
    //　乱数系
    struct Rand{
        std::mt19937 mt_rand;
        virtual std::vector<int> row_sampling(int sample_num, int rows);
        virtual std::vector<int> col_sampling(int sample_num, int cols);
    };
    
private:
    // 決定木
    class RF_CLF_tree:public CLF_tree{
    protected:
        Tree_Split make_split(std::vector<int>& data_idx);
        Random_Forest_clf* forest_pointer;
        
    public:
        //RF_CLF_tree(CLF_tree_param& param,Random_Forest_clf* forest_address);
        RF_CLF_tree();
        void set_forest(Random_Forest_clf* forest_address);
        void fit(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, std::vector<int>& data_idx);
        
        //テストクラスのフレンド宣言
        friend class RandomForestTest;
        FRIEND_TEST(RandomForestTest, make_split);
        FRIEND_TEST(RandomForestTest, fit);

        
    };
   

//メンバの定義
protected:
    std::vector<RF_CLF_tree> forest;
    Random_Forest_clf_param param;
    const Eigen::MatrixXd* X_train;
    const Eigen::MatrixXd* y_train;
    
    // 乱数系
    Rand* rand;
    Rand rand_instance;

public:
    Random_Forest_clf();
    Random_Forest_clf(Random_Forest_clf_param& param);
    void set_param(Random_Forest_clf_param& param);
    void fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X_test);

    //テストクラスのフレンド宣言
    friend class RandomForestTest;
    FRIEND_TEST(RandomForestTest, row_sampling);
    FRIEND_TEST(RandomForestTest, col_sampling);
    FRIEND_TEST(RandomForestTest, make_split);
    FRIEND_TEST(RandomForestTest, fit);
    FRIEND_TEST(RandomForestTest, predict);
    
/*
public:
    class test{
    public:
        test(){cout<<"succsess!!"<<endl;}
    };
*/
    
};


#endif /* random_forest_hpp */
