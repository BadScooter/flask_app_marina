import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import json

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'

    @app.route('/', methods=['GET', 'POST'])
    def XGBoost():
        plot_png1, plot_png2, rmse = None, None, None
        if(request.args.get('plot1')):
            plot_png1 = request.args.get('plot1')
            plot_png2 = request.args.get('plot2')
            rmse = request.args.get('metric')

        if request.method == "POST":
            
            feats = request.form.getlist('features')
            feats.append('arrivals')
            
            nums = request.form.get('numlags')
            if nums == None: nums = 0
                
            opt = request.form.get('inlineRadioOptions')

            if(int(nums) < 1 and len(feats) == 1 and opt == 'manual'): 
                flash('Select some features.', category='error')
            else:
                from xgboost_web import xgboost_func    
                plot_png1, plot_png2, rmse = xgboost_func(features=feats, numlags=int(nums), option=opt)
                return redirect(url_for('XGBoost', plot1=plot_png1, plot2=plot_png2, metric=rmse))
        
        return render_template("XGBoost.html", plot_url1=plot_png1, plot_url2=plot_png2, metric_url=rmse)
    return app

app = create_app()

#if __name__ == '__main__':
#    app.run(debug=True)