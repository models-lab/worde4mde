
## Running the local server

```
python manage.py runserver
```

## Deploy

To run the page behind a reverse proxy (choose secret and port):

```
DJANGO_SECRET="1234abc" python manage.py runserver 0.0.0.0:8081
```

Also, you need to adjust the relative paths of the server according
to the reverse proxy configuration:

```
export SERVER_PREFIX='worde4mde/'
```

Also, set debugging of:

```
export DJANGO_DEBUG=False
```

For example, in HAProxy we can force a beginning of the path like:

```
use_backend app-worde4mde if { path /worde4mde } || { path_beg /worde4mde/ }
```  