let auth0ClientPromise;

function getAuth0Client() {
  if (!auth0ClientPromise) {
    auth0ClientPromise = auth0.createAuth0Client({
      domain: "login.fotosort.com",
      clientId: "t8MAyGQPxUK0k673gQc7O8zhf6XtrZoD",
      cacheLocation: "localstorage",
      authorizationParams: {
        redirect_uri: window.location.origin + "/console.html",
        audience: "https://9jy8ljy0k0.execute-api.us-east-1.amazonaws.com"
      }
    });
  }
  return auth0ClientPromise;
}

async function getAccessToken() {
  const client = await getAuth0Client();
  return client.getTokenSilently();
}

async function getBearer() {
  return `Bearer ${await getAccessToken()}`;
}
