# postgres login details

username = ''
password = ''
dbname = ''


SQLALCHEMY_DATABASE_URI = 'postgresql://%s:%s@localhost/%s' % (username, password, dbname)
