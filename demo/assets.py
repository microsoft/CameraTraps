from flask import Flask
from flask_assets import Bundle
from . import app, webassets


js_all = Bundle("node_modules/jquery/dist/jquery.min.js",
                 "js/popper.min.js",
                 "node_modules/bootstrap/dist/js/bootstrap.min.js",
                 "node_modules/noty/lib/noty.js",
                 "js/jquery.waterwheelCarousel.min.js",
                 "js/blazy.min.js",
                 "js/camera-trap.js",
                 filters="jsmin",
                 output="js/libs.js")


css_all = Bundle(Bundle("node_modules/bootstrap/dist/css/bootstrap.css"),
                  Bundle("css/camera-trap.css"),
                  Bundle("node_modules/noty/lib/noty.css"),
                  filters="cssmin",
                  output="css/main.css")


webassets.register('js_all', js_all)
webassets.register('css_all', css_all)

