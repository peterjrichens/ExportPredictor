# postgres login details

username = 'peter'
password = '1910'
dbname = 'peter'


SQLALCHEMY_DATABASE_URI = 'postgresql://%s:%s@localhost/%s' % (username, password, dbname)
