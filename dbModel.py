from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)

class Country(db.Model):
    __tablename__ = "countries"
    code = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode(80))
    iso2 = db.Column(db.String(2))
    iso = db.Column(db.String(3))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    region = db.Column(db.Unicode(30))
    income_level = db.Column(db.Unicode(30))


    def __init__(self, code, name, iso, latitude, longitude, region, income_level):
        self.code = code
        self.name = name
        self.iso2 = iso
        self.iso = iso
        self.latitude = latitude
        self.longitude = longitude
        self.region = region
        self.income_level = income_level


    def __repr__(self):
        return '<Country %r>' % self.name

    @property
    def serialize(self):
       """Return object data in easily serializeable format"""
       return {
           'code': self.code,
           'name': self.name,
           'iso2': self.iso2,
           'iso': self.iso,
           'latitude': self.latitude,
           'longitude': self.longitude,
           'region': self.region,
           'income_level': self.income_level
       }

class Commodity(db.Model):
    __tablename__ = "commodities"
    code = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode(260))
    code_2dg = db.Column(db.Integer)
    name_2dg = db.Column(db.Unicode(260))
    name_1dg = db.Column(db.Unicode(260))


    def __init__(self, code, name, code_2dg, name_2dg, name_1dg):
        self.code = code
        self.name = name
        self.code_2dg = code_2dg
        self.name_2dg = name_2dg
        self.name_1dg = name_1dg

    def __repr__(self):
        return '<Commodity %r>' % self.name

class Comtrade(db.Model):
    __tablename__ = "comtrade"
    origin = db.Column(db.Integer, db.ForeignKey('countries.code'), primary_key=True)
    destination = db.Column(db.Integer, primary_key=True)
    countries = db.relationship('Country', backref=db.backref('Country', lazy='dynamic'))

    cmd = db.Column(db.Integer, db.ForeignKey('commodities.code'), primary_key=True)
    commodities = db.relationship('Commodity', backref=db.backref('Commodity', lazy='dynamic'))

    yr = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.BigInteger)

    def __init__(self, origin, destination, cmd, yr, value):
        self.origin = origin
        self.destination = destination
        self.cmd = cmd
        self.yr = yr
        self.value = value

class MLDataset(db.Model):
    __tablename__ = "mldataset"
    origin = db.Column(db.Integer, db.ForeignKey('countries.code'), primary_key=True)
    cmd = db.Column(db.Integer, db.ForeignKey('commodities.code'), primary_key=True)
    year = db.Column(db.Integer, primary_key=True)
    countries = db.relationship('Country', backref=db.backref('Country_ml', lazy='dynamic'), foreign_keys=[origin, cmd, year])
    commodities = db.relationship('Commodity', backref=db.backref('Commodity_ml', lazy='dynamic'), foreign_keys=[origin, cmd, year])
    new_export = db.Column(db.Integer)
    rca = db.Column(db.Float)
    export_destination = db.Column(db.Float)
    intensity = db.Column(db.Float)
    imports = db.Column(db.Float)
    import_origin = db.Column(db.Float)
    market_share = db.Column(db.Float)
    distance = db.Column(db.Float)
    origin_average = db.Column(db.Float)
    cmd_average = db.Column(db.Float)

    def __init__(self, origin, cmd, year, new_export, rca, export_destination, intensity,
                 imports, import_origin, market_share, distance, origin_average, cmd_average):
        self.origin = origin
        self.cmd = cmd
        self.year = year
        self.new_export = new_export
        self.rca = rca
        self.export_destination = export_destination
        self.intensity = intensity
        self.imports = imports
        self.import_origin = import_origin
        self.market_share = market_share
        self.distance = distance
        self.origin_average = origin_average
        self.cmd_average = cmd_average

# database for an alternative definition of 'new export'
class MLDataset2(db.Model):
    __tablename__ = "mldataset2"
    origin = db.Column(db.Integer, db.ForeignKey('mldataset.origin'), primary_key=True)
    cmd = db.Column(db.Integer, db.ForeignKey('mldataset.cmd'), primary_key=True)
    year = db.Column(db.Integer, db.ForeignKey('mldataset.year'), primary_key=True)
    # new names for relationship class variables
    countries = db.relationship('Country', backref=db.backref('Country_ml2', lazy='dynamic'), foreign_keys=[origin, cmd, year])
    commodities = db.relationship('Commodity', backref=db.backref('Commodity_ml2', lazy='dynamic'), foreign_keys=[origin, cmd, year])
    new_export = db.Column(db.Integer)
    rca = db.Column(db.Float)
    export_destination = db.Column(db.Float)
    intensity = db.Column(db.Float)
    imports = db.Column(db.Float)
    import_origin = db.Column(db.Float)
    market_share = db.Column(db.Float)
    distance = db.Column(db.Float)
    origin_average = db.Column(db.Float)
    cmd_average = db.Column(db.Float)

    def __init__(self, origin, cmd, year, new_export, rca, export_destination, intensity,
                 imports, import_origin, market_share, distance, origin_average, cmd_average):
        self.origin = origin
        self.cmd = cmd
        self.year = year
        self.new_export = new_export
        self.rca = rca
        self.export_destination = export_destination
        self.intensity = intensity
        self.imports = imports
        self.import_origin = import_origin
        self.market_share = market_share
        self.distance = distance
        self.origin_average = origin_average
        self.cmd_average = cmd_average



