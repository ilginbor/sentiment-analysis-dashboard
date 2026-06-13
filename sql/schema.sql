IF DB_ID('SentimentDashboardDB') IS NULL
BEGIN
    CREATE DATABASE SentimentDashboardDB;
END
GO

USE SentimentDashboardDB;
GO

IF OBJECT_ID('reviews', 'U') IS NULL
BEGIN
    CREATE TABLE reviews (
        review_id INT IDENTITY(1,1) PRIMARY KEY,
        review_text NVARCHAR(MAX) NOT NULL,
        actual_sentiment NVARCHAR(20) NULL,
        predicted_sentiment NVARCHAR(20) NULL,
        confidence_score FLOAT NULL,
        classification_time DATETIME2 DEFAULT SYSDATETIME()
    );
END
GO

IF OBJECT_ID('model_results', 'U') IS NULL
BEGIN
    CREATE TABLE model_results (
        result_id INT IDENTITY(1,1) PRIMARY KEY,
        model_name NVARCHAR(100) NOT NULL,
        accuracy FLOAT NOT NULL,
        precision_score FLOAT NOT NULL,
        recall_score FLOAT NOT NULL,
        f1_score FLOAT NOT NULL,
        created_at DATETIME2 DEFAULT SYSDATETIME()
    );
END
GO

IF OBJECT_ID('prepared_reviews', 'U') IS NULL
BEGIN
    CREATE TABLE prepared_reviews (
        prepared_review_id INT IDENTITY(1,1) PRIMARY KEY,
        review_text NVARCHAR(MAX) NOT NULL,
        cleaned_text NVARCHAR(MAX) NOT NULL,
        tokenized_text NVARCHAR(MAX) NOT NULL,
        sentiment NVARCHAR(20) NOT NULL,
        created_at DATETIME2 DEFAULT SYSDATETIME()
    );
END
GO

IF OBJECT_ID('model_confusion_matrices', 'U') IS NULL
BEGIN
    CREATE TABLE model_confusion_matrices (
        matrix_id INT IDENTITY(1,1) PRIMARY KEY,
        model_name NVARCHAR(100) NOT NULL,
        actual_sentiment NVARCHAR(20) NOT NULL,
        predicted_sentiment NVARCHAR(20) NOT NULL,
        count_value INT NOT NULL,
        created_at DATETIME2 DEFAULT SYSDATETIME()
    );
END
GO