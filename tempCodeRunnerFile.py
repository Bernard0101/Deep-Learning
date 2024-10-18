def Calculate_Loss(self, target, prediction, function):
        if(function == "MSE"):
            MSE_Loss=nn_func.Loss_MSE(y_pred=prediction, y_label=target)
            return MSE_Loss
        if (function == "MAE"):
            MAE_Loss=nn_func.Loss_MAE(y_pred=prediction, y_label=target)
            return MAE_Loss